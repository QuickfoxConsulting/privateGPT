

"""Website crawling service for content ingestion and scheduling."""
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, AsyncUrlSeeder, SeedingConfig
from crawl4ai.async_configs import BrowserConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from private_gpt.users.models.integration import WebsiteCrawlConfig, WebsiteCrawlPage
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.di import global_injector
from private_gpt.server.integrations.base import BaseIntegration, IntegrationMetadata

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = BackgroundScheduler()


class WebsiteCrawlService(BaseIntegration):
    """Service for website crawling, page management, and content ingestion."""
    
    def __init__(self, db: Session):
        BaseIntegration.__init__(self)
        self.db = db

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            id="website_crawl",
            name="Website Crawler",
            description="Crawl and ingest content from websites.",
            icon="globe",  # Frontend mapping
            is_configurable=True,
            settings_path="/admin/website-crawl"
        )

    def is_installed(self, user_id: int) -> bool:
        """Check if integration is active in the registry."""
        from private_gpt.users.models.integration import Integration
        record = self.db.query(Integration).filter(
            Integration.user_id == user_id,
            Integration.integration_id == self.metadata.id,
            Integration.is_active == True
        ).first()
        return bool(record)

    def install(self, user_id: int) -> bool:
        """
        No specific setup needed for Website Crawl until a config is created.
        We return True to indicate success.
        """
        return True

    def uninstall(self, user_id: int) -> bool:
        """Delete all configs for user."""
        configs = self.list_configs(user_id)
        for config in configs:
            # Cleanup runs in sync context here, wrapping async inside if needed
            # But delete_config is async. This might be tricky.
            # Ideally uninstall is async or we use run_until_complete
            pass 
            # We will rely on manual deletion for now or implement full cleanup loop
        return True

    def get_config(self, user_id: int) -> Optional[Dict[str, Any]]:
        # Return summary stats
        configs = self.list_configs(user_id)
        return {
            "config_count": len(configs),
            "total_pages": sum(c.pages_crawled or 0 for c in configs)
        }

    def create_config(
        self,
        user_id: int,
        name: str,
        base_url: str,
        max_depth: int = 3,
        url_patterns: Optional[Dict] = None,
        crawl_frequency: Optional[str] = None,
        use_sitemap: bool = True,
        seeding_strategy: str = "smart"  # "smart", "sitemap", "bfs", "dfs"
    ) -> WebsiteCrawlConfig:
        """Create a new website crawl configuration."""
        config = WebsiteCrawlConfig(
            user_id=user_id,
            name=name,
            base_url=base_url,
            max_depth=max_depth,
            url_patterns=url_patterns or {"include": [], "exclude": []},
            crawl_frequency=crawl_frequency,
            enabled=True
        )
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        
        # Schedule if frequency is set
        if crawl_frequency:
            self.schedule_crawl(config.id, crawl_frequency)
        
        return config

    def get_config(self, config_id: int, user_id: int) -> Optional[WebsiteCrawlConfig]:
        """Get a specific crawl configuration."""
        return self.db.query(WebsiteCrawlConfig).filter(
            WebsiteCrawlConfig.id == config_id,
            WebsiteCrawlConfig.user_id == user_id
        ).first()

    def list_configs(self, user_id: int) -> List[WebsiteCrawlConfig]:
        """List all crawl configurations for a user."""
        return self.db.query(WebsiteCrawlConfig).filter(
            WebsiteCrawlConfig.user_id == user_id
        ).all()

    def update_config(
        self,
        config_id: int,
        user_id: int,
        **kwargs
    ) -> Optional[WebsiteCrawlConfig]:
        """Update a crawl configuration."""
        config = self.get_config(config_id, user_id)
        if not config:
            return None
        
        old_frequency = config.crawl_frequency
        new_frequency = kwargs.get("crawl_frequency")
        
        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        config.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(config)
        
        # Update schedule if needed
        if "crawl_frequency" in kwargs:
            if new_frequency:
                self.schedule_crawl(config.id, new_frequency)
            else:
                self.remove_schedule(config.id)
        
        return config

    async def delete_config(self, config_id: int, user_id: int) -> bool:
        """Delete a crawl configuration and associated ingested documents."""
        config = self.get_config(config_id, user_id)
        if config:
            # 1. Cleanup documents from vector store
            ingest_service = global_injector.get(IngestService)
            
            # First try bulk deletion by config_id (for future/newly ingested docs)
            await ingest_service.delete_by_metadata("website_config_id", str(config.id))
            
            # Also cleanup by URL for older ingested docs that don't have the config_id tagged
            pages = self.db.query(WebsiteCrawlPage).filter(
                WebsiteCrawlPage.config_id == config.id
            ).all()
            
            for page in pages:
                if page.ingest_status == "ingested":
                    try:
                        logger.info(f"Removing document for URL {page.url} from vector store")
                        await ingest_service.delete_by_metadata("url", page.url)
                    except Exception as e:
                        logger.warning(f"Failed to delete vector nodes for {page.url}: {str(e)}")

            # 2. Remove schedule and DB records
            self.remove_schedule(config.id)
            self.db.delete(config)
            self.db.commit()
            return True
        return False

    def list_pages(
        self,
        config_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        filter_status: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> Tuple[List[WebsiteCrawlPage], int]:
        """List discovered pages for a configuration."""
        config = self.get_config(config_id, user_id)
        if not config:
            return [], 0
        
        query = self.db.query(WebsiteCrawlPage).filter(
            WebsiteCrawlPage.config_id == config_id
        )

        if filter_status and filter_status != "all":
            query = query.filter(WebsiteCrawlPage.ingest_status == filter_status)
            
        if search_query:
            search = f"%{search_query}%"
            query = query.filter(
                (WebsiteCrawlPage.url.ilike(search)) | 
                (WebsiteCrawlPage.title.ilike(search))
            )

        total = query.count()
        items = query.offset(skip).limit(limit).all()
        return items, total

    def update_page_selection(
        self,
        config_id: int,
        user_id: int,
        page_ids: List[int],
        is_selected: bool
    ) -> bool:
        """Update selection status for pages."""
        config = self.get_config(config_id, user_id)
        if not config:
            return False
        
        self.db.query(WebsiteCrawlPage).filter(
            WebsiteCrawlPage.config_id == config_id,
            WebsiteCrawlPage.id.in_(page_ids)
        ).update({"is_selected": is_selected}, synchronize_session=False)
        
        self.db.commit()
        return True


    def get_ingested_pages(self, department_id: int) -> List[WebsiteCrawlPage]:
        from private_gpt.users.models.user import User
        """Get all ingested and selected pages for a department across all configs."""
        return self.db.query(WebsiteCrawlPage).join(
            WebsiteCrawlConfig,
            WebsiteCrawlPage.config_id == WebsiteCrawlConfig.id
        ).join(
            User,
            WebsiteCrawlConfig.user_id == User.id
        ).filter(
            User.department_id == department_id,
            WebsiteCrawlConfig.enabled == True,
            WebsiteCrawlPage.is_selected == True,
            WebsiteCrawlPage.ingest_status == "ingested"
        ).all()

    def should_crawl_url(self, url: str, url_patterns: Dict, base_url: str) -> bool:
        """Check if a URL should be crawled based on patterns and asset filtering."""
        # Ensure URL is from the same domain
        parsed_url = urlparse(url)
        base_domain = urlparse(base_url).netloc
        url_domain = parsed_url.netloc
        
        if url_domain != base_domain:
            return False
        
        # Exclude common assets and irrelevant paths
        path = parsed_url.path.lower()
        
        # Extensions to exclude
        exclude_extensions = {
            '.css', '.js', '.json', '.xml', '.png', '.jpg', '.jpeg', '.gif', 
            '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.mp4', '.mp3', 
            '.pdf', '.zip', '.gz', '.tgz', '.log', '.logs', '.map'
        }
        if any(path.endswith(ext) for ext in exclude_extensions):
            return False
            
        # Path segments to exclude
        exclude_segments = {
            '/css/', '/js/', '/assets/', '/static/', '/wp-includes/', 
            '/wp-content/plugins/', '/wp-content/themes/', '/logs/', 
            '/cache/', '/tmp/', '/vendor/'
        }
        if any(segment in path for segment in exclude_segments):
            return False
        
        include_patterns = url_patterns.get("include", [])
        exclude_patterns = url_patterns.get("exclude", [])
        
        # Check exclude patterns first
        for pattern in exclude_patterns:
            if pattern in url:
                return False
        
        # If no include patterns, allow all (except excluded)
        if not include_patterns:
            return True
        
        # Check include patterns
        return any(pattern in url for pattern in include_patterns)

    async def _parse_sitemap_xml(self, xml_content: str) -> Set[str]:
        """Parse XML sitemap and extract all URLs. Handles multiple formats and malformed XML."""
        urls = set()
        if not xml_content:
            return urls

        # 1. Primary extraction: Use regex (very robust for sitemaps/feeds)
        import re
        
        # Extract from <loc> tags (Sitemaps)
        loc_urls = re.findall(r'<loc>\s*(https?://[^<>\s"\' ]+)\s*</loc>', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in loc_urls if u.strip())
        
        # Extract from <link> tags (RSS/Atom)
        # Handle <link>URL</link>
        link_tag_urls = re.findall(r'<link>\s*(https?://[^<>\s"\' ]+)\s*</link>', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in link_tag_urls if u.strip())
        
        # Handle <link href="URL" ... />
        link_href_urls = re.findall(r'<link[^>]+href=["\'](https?://[^"\'\s>]+)["\']', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in link_href_urls if u.strip())

        # 2. Secondary extraction: XML parsing (more precise if it works)
        try:
            # Clean up content for XML parser
            # Strip ALL namespaces to make findall work without namespaces
            cleaned_xml = re.sub(r'\s+xmlns(:\w+)?=["\'][^"\']+["\']', '', xml_content)
            # Remove XML declaration if it causes issues
            cleaned_xml = re.sub(r'<\?xml[^?]+\?>', '', cleaned_xml).strip()
            
            if cleaned_xml:
                # Add a dummy root if it looks like a list of fragments
                if not cleaned_xml.startswith('<'):
                    cleaned_xml = f"<root>{cleaned_xml}</root>"
                
                try:
                    root = ET.fromstring(cleaned_xml)
                    
                    # Search for any 'loc' or 'link' tags regardless of depth
                    for elem in root.iter():
                        tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
                        if tag == 'loc':
                            if elem.text and elem.text.strip():
                                urls.add(elem.text.strip())
                        elif tag == 'link':
                            # check text
                            if elem.text and elem.text.strip():
                                urls.add(elem.text.strip())
                            # check href attribute
                            href = elem.get('href')
                            if href and href.strip():
                                urls.add(href.strip())
                except ET.ParseError:
                    # If XML parsing fails, we already have regex results
                    pass
                    
        except Exception:
            # Silence all secondary parsing errors since we have regex fallback
            pass
        
        if urls:
            logger.info(f"Discovered {len(urls)} URLs from sitemap/feed content")
        
        return urls
    
    def _extract_urls_with_regex(self, content: str) -> Set[str]:
        """Fallback method to extract URLs using regex when XML parsing fails."""
        import re
        urls = set()
        
        # Extract URLs from <loc> tags using regex
        loc_pattern = r'<loc>\s*(https?://[^<\s]+)\s*</loc>'
        matches = re.findall(loc_pattern, content, re.IGNORECASE)
        
        for url in matches:
            cleaned_url = url.strip()
            if cleaned_url:
                urls.add(cleaned_url)
        
        # Also try to find URLs in <link> tags (RSS/Atom feeds)
        link_pattern = r'<link[^>]*(?:href=["\'](https?://[^"\']+)["\']|>(https?://[^<]+)</link>)'
        link_matches = re.findall(link_pattern, content, re.IGNORECASE)
        
        for match in link_matches:
            # match might be a tuple if there are groups
            if isinstance(match, tuple):
                for url in match:
                    if url and url.strip():
                        urls.add(url.strip())
            elif match and match.strip():
                urls.add(match.strip())
        
        return urls

    async def _fetch_sitemap_urls(
        self,
        base_url: str,
        max_urls: int = 100
    ) -> Set[str]:
        """Fetch and parse sitemap(s) using crawl4ai's AsyncUrlSeeder."""
        all_urls = set()
        
        try:
            logger.info(f"Using AsyncUrlSeeder for {base_url}")
            async with AsyncUrlSeeder() as seeder:
                config = SeedingConfig(
                    source="sitemap+cc",  # Use sitemap and Common Crawl for maximum coverage
                    max_urls=max_urls
                )
                
                # Extract domain for seeder
                domain = urlparse(base_url).netloc
                if not domain:
                    domain = base_url
                
                discovered_urls = await seeder.urls(domain, config)
                
                for item in discovered_urls:
                    if isinstance(item, dict) and "url" in item:
                        all_urls.add(item["url"])
                    elif isinstance(item, str):
                        all_urls.add(item)
                        
            logger.info(f"AsyncUrlSeeder discovered {len(all_urls)} URLs")
        except Exception as e:
            logger.error(f"AsyncUrlSeeder failed: {str(e)}")
            # Fallback to a very basic sitemap.xml fetch if seeder fails
            try:
                sitemap_url = urljoin(base_url, "/sitemap.xml")
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=sitemap_url)
                    if result.success and result.html:
                        urls = await self._parse_sitemap_xml(result.html)
                        all_urls.update(urls)
            except Exception as se:
                logger.error(f"Manual sitemap fallback failed: {str(se)}")
                
        return all_urls

    async def scan_website(
        self,
        config: WebsiteCrawlConfig,
        max_pages: int = 100,
        use_sitemap: bool = True
    ) -> List[WebsiteCrawlPage]:
        """
        Scan website to discover URLs using crawl4ai's URL seeding strategies.
        This leverages sitemap crawling and proper BFS/DFS strategies.
        """
        config.crawl_status = "scanning"
        self.db.commit()
        
        try:
            # Get existing pages to avoid duplicates
            existing_urls = {
                page.url for page in self.db.query(WebsiteCrawlPage).filter(
                    WebsiteCrawlPage.config_id == config.id
                ).all()
            }
            
            discovered_pages = []
            
            # Configure browser
            browser_config = BrowserConfig(
                headless=True,
                verbose=True
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Create run configuration
                prune_filter = PruningContentFilter(
                    threshold=0.45,           
                    threshold_type="dynamic",  
                    min_word_threshold=5      
                )
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,  
                    markdown_generator=DefaultMarkdownGenerator(content_filter=prune_filter),    
                )
                
                # Discover URLs
                urls_to_process = set()
                
                if use_sitemap:
                    logger.info(f"Discovering URLs using AsyncUrlSeeder for {config.base_url}")
                    sitemap_urls = await self._fetch_sitemap_urls(
                        base_url=config.base_url,
                        max_urls=max_pages * 2  # Find a few more to filter
                    )
                    urls_to_process.update(sitemap_urls)
                
                # If sitemap didn't yield results, use BFS
                if not urls_to_process:
                    logger.info("No sitemap URLs found, using BFS crawling")
                    urls_to_process = await self._bfs_crawl(
                        crawler=crawler,
                        start_url=config.base_url,
                        max_depth=config.max_depth,
                        max_pages=max_pages,
                        run_config=run_config
                    )
                
                # Process discovered URLs
                processed_count = 0
                for url in urls_to_process:
                    if processed_count >= max_pages:
                        break
                    
                    if url in existing_urls:
                        continue
                    
                    if not self.should_crawl_url(url, config.url_patterns, config.base_url):
                        continue
                    
                    try:
                        # Fetch page to get title
                        result = await crawler.arun(url=url, config=run_config)
                        
                        if result.success:
                            title = ""
                            if hasattr(result, 'metadata') and result.metadata:
                                title = result.metadata.get("title", "")
                            
                            new_page = WebsiteCrawlPage(
                                config_id=config.id,
                                url=url,
                                title=title,
                                is_selected=True,
                                ingest_status="pending"
                            )
                            self.db.add(new_page)
                            discovered_pages.append(new_page)
                            existing_urls.add(url)
                            processed_count += 1
                            
                            logger.info(f"Discovered page {processed_count}/{max_pages}: {url}")
                            
                            # Commit in batches to avoid memory issues
                            if processed_count % 20 == 0:
                                self.db.commit()
                    
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
                        continue
                
                self.db.commit()
                config.crawl_status = "completed"
                config.pages_crawled = len(discovered_pages)
                self.db.commit()
                
                return discovered_pages
        
        except Exception as e:
            logger.error(f"Scan failed for config {config.id}: {str(e)}")
            config.crawl_status = "failed"
            self.db.commit()
            raise

    async def _bfs_crawl(
        self,
        crawler: AsyncWebCrawler,
        start_url: str,
        max_depth: int,
        max_pages: int,
        run_config: CrawlerRunConfig
    ) -> Set[str]:
        """
        Perform BFS crawling to discover URLs.
        Returns a set of discovered URLs.
        """
        discovered_urls = set()
        queue = [(start_url, 0)]  # (url, depth)
        visited = set()
        
        while queue and len(discovered_urls) < max_pages:
            current_url, depth = queue.pop(0)
            
            if current_url in visited or depth > max_depth:
                continue
            
            visited.add(current_url)
            discovered_urls.add(current_url)
            
            try:
                result = await crawler.arun(url=current_url, config=run_config)
                
                if result.success and depth < max_depth:
                    # Extract links for next level
                    if hasattr(result, 'links') and result.links:
                        for link in result.links:
                            href = link.get('href') if isinstance(link, dict) else getattr(link, 'href', None)
                            
                            if href and href.startswith('http') and href not in visited:
                                # Check if same domain
                                if urlparse(href).netloc == urlparse(start_url).netloc:
                                    queue.append((href, depth + 1))
            
            except Exception as e:
                logger.debug(f"Error crawling {current_url}: {str(e)}")
                continue
        
        return discovered_urls

    async def ingest_selected_pages(self, config: WebsiteCrawlConfig) -> int:
        """Ingest content from selected pages using AsyncWebCrawler."""
        config.crawl_status = "ingesting"
        self.db.commit()
        
        ingest_service = global_injector.get(IngestService)
        
        pages_to_ingest = self.db.query(WebsiteCrawlPage).filter(
            WebsiteCrawlPage.config_id == config.id,
            WebsiteCrawlPage.is_selected == True
        ).all()
        
        count = 0
        
        browser_config = BrowserConfig(headless=True, verbose=True)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            prune_filter = PruningContentFilter(
                threshold=0.45,           
                threshold_type="dynamic",  
                min_word_threshold=5      
            )
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,  
                markdown_generator=DefaultMarkdownGenerator(content_filter=prune_filter),    
            )
            
            for page in pages_to_ingest:
                try:
                    # Remove existing data for this URL before re-ingesting
                    logger.info(f"Removing old data for {page.url} before re-ingestion")
                    await ingest_service.delete_by_metadata("url", page.url)
                    
                    result = await crawler.arun(url=page.url, config=run_config)
                    
                    if result.success and result.markdown:
                        # Sanitize filename from URL
                        import re
                        safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', page.url)
                        # Ensure it doesn't start with '..'
                        if safe_filename.startswith('..'):
                            safe_filename = '_' + safe_filename
                        if len(safe_filename) > 255:
                            safe_filename = safe_filename[-255:]

                        # Ingest content
                        await ingest_service.ingest_text(
                            file_name=safe_filename,
                            text=result.markdown,
                            metadata={
                                "type": "website",
                                "url": page.url,
                                "title": page.title or "",
                                "website_config_id": str(config.id)
                            }
                        )
                        
                        page.ingest_status = "ingested"
                        page.last_ingested_at = datetime.utcnow()
                        count += 1
                    else:
                        page.ingest_status = "error"
                        error_msg = getattr(result, 'error_message', 'Unknown error')
                        page.error_message = f"Failed to crawl: {error_msg}"
                
                except Exception as e:
                    logger.error(f"Ingestion failed for {page.url}: {str(e)}")
                    page.ingest_status = "error"
                    page.error_message = str(e)
                
                self.db.commit()
        
        config.crawl_status = "completed"
        config.last_crawl = datetime.utcnow()
        config.pages_crawled = count
        self.db.commit()
        
        return count

    # --- Scheduling ---
    
    @staticmethod
    def start_scheduler():
        """Start the global scheduler."""
        if not scheduler.running:
            scheduler.start()
            logger.info("Website Crawl Scheduler started")

    def schedule_crawl(self, config_id: int, cron_expression: str):
        """Schedule a crawl job."""
        job_id = f"crawl_{config_id}"
        
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
        
        try:
            scheduler.add_job(
                self._run_scheduled_ingest,
                CronTrigger.from_crontab(cron_expression),
                id=job_id,
                args=[config_id],
                replace_existing=True
            )
            logger.info(f"Scheduled crawl for config {config_id} with cron {cron_expression}")
        except Exception as e:
            logger.error(f"Failed to schedule crawl for {config_id}: {str(e)}")

    def remove_schedule(self, config_id: int):
        """Remove a crawl job."""
        job_id = f"crawl_{config_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

    @staticmethod
    def _run_scheduled_ingest(config_id: int):
        """Wrapper to run scheduled ingest in a new DB session."""
        import asyncio
        from private_gpt.users.db.session import SessionLocal
        
        db = SessionLocal()
        service = WebsiteCrawlService(db)
        
        try:
            config = db.query(WebsiteCrawlConfig).filter(
                WebsiteCrawlConfig.id == config_id
            ).first()
            
            if config:
                logger.info(f"Starting scheduled ingest for {config.name}")
                asyncio.run(service.ingest_selected_pages(config))
        except Exception as e:
            logger.error(f"Scheduled task failed: {str(e)}")
        finally:
            db.close()