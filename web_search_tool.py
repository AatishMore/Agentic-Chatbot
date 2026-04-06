

from langchain_core.tools import tool
from ddgs import DDGS


@tool
def web_search(query: str) -> str:
    """
    Search the internet using DuckDuckGo.
    Returns raw web snippets — no formatting, no LLM.
    Use this tool when the answer requires external or recent knowledge
    not available in internal documents.
    Do NOT use for internal docs, company policies, or uploaded PDFs.
    """
    print(f"[WEB TOOL] Searching DuckDuckGo for: {query}")
    try:
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                body = r.get("body", "").strip()
                title = r.get("title", "").strip()
                href = r.get("href", "").strip()
                if body:
                    # Include title + url as lightweight metadata for manager context
                    entry = f"[{title}]({href})\n{body}" if title else body
                    snippets.append(entry)

        if not snippets:
            return "NO_RESULTS"

        # Return raw snippets separated by a clear delimiter — manager will format
        return "\n\n---SNIPPET---\n\n".join(snippets)

    except Exception as e:
        print(f"[WEB TOOL ERROR] {e}")
        return "NO_RESULTS"
