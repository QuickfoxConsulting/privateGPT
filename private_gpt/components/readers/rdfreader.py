
import logging
from pathlib import Path
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger(__name__)

class RDFReader(BaseReader):
    """RDF reader."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize loader."""
        super().__init__(*args, **kwargs)

    def fetch_labels(self, uri: URIRef, graph: Graph, lang: str):
        """Fetch all labels of a URI by language."""
        return list(
            filter(lambda x: x.language in [lang, None], graph.objects(uri, RDFS.label))
        )

    def fetch_label_in_graphs(self, uri: URIRef, lang: str = "en"):
        """Fetch one label of a URI by language from the local or global graph."""
        labels = self.fetch_labels(uri, self.g_local, lang)
        if len(labels) > 0:
            return labels[0].value

        labels = self.fetch_labels(uri, self.g_global, lang)
        if len(labels) > 0:
            return labels[0].value

        return str(uri)

    def load_data(self, file: Path, extra_info: dict | None = None) -> list[Document]:
        """Parse file."""
        extra_info = extra_info or {}
        extra_info["graph_type"] = "rdf"
        lang = (
            extra_info["lang"]
            if extra_info is not None and "lang" in extra_info
            else "en"
        )

        self.g_local = Graph()
        self.g_local.parse(file)

        self.g_global = Graph()
        self.g_global.parse(str(RDF))
        self.g_global.parse(str(RDFS))

        text_list = []

        for s, p, o in self.g_local:
            logger.debug("s=%s, p=%s, o=%s", s, p, o)
            if p == RDFS.label:
                continue

            subj_label = self.fetch_label_in_graphs(s, lang=lang)
            pred_label = self.fetch_label_in_graphs(p, lang=lang)
            obj_label = self.fetch_label_in_graphs(o, lang=lang)

            if subj_label is None or pred_label is None or obj_label is None:
                continue

            triple = f"<{subj_label}> " f"<{pred_label}> " f"<{obj_label}>"
            text_list.append(triple)

        text = "\n".join(text_list)
        return [self._text_to_document(text, extra_info)]

    def _text_to_document(self, text: str, extra_info: dict | None = None) -> Document:
        return Document(text=text, extra_info=extra_info or {})