"""
Directives for the VLM-based Raw Transcription Engine.
"""

VLM_OCR_COT_PROMPT = """
You are trained ro perform ocr on a page. Which is really important and cannot be mistaken.
perform ocr on a page. In such way it is like human reading style , so things can related to each other . try understanding so its like reading and writting at a same time like a human
and for answer just provide the raw text.
"""

VLM_OCR_SIMPLE_PROMPT = """
Transcribe all text in this image exactly. No chatter or labels. Output only the raw text.
"""
