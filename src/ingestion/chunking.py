
import re
import hashlib


def chunk_records(
    document_id: str,
    filename: str,
    records: list[dict],
    chunk_size: int = 1200,
    overlap: int = 100,
) -> list[dict]:
    if chunk_size < 50 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Require chunk_size >= 50 and 0 <= overlap < chunk_size"
        )

    chunks = []

    for record in records:
        text = record["text"].strip()
        if not text:
            continue

        # Preserve paragraph and table-row boundaries.
        lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

        current = ""
        parts = []

        for line in lines:
            candidate = f"{current}\n{line}" if current else line

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    parts.append(current)

                # Keep long lines intact where possible.
                current = line

                if len(current) > chunk_size:
                    parts.append(current[:chunk_size])
                    current = current[chunk_size:]

        if current:
            parts.append(current)

        for index, part in enumerate(parts):
            if index > 0 and overlap:
                previous = parts[index - 1]
                prefix = previous[-overlap:]
                part = prefix + "\n" + part
                part = part[:chunk_size]

            raw_id = (
                f"{document_id}:{record.get('page')}:"
                f"{index}:{part}"
            )

            chunks.append({
                "chunk_id": hashlib.sha1(
                    raw_id.encode("utf-8")
                ).hexdigest()[:12],
                "document_id": document_id,
                "filename": filename,
                "page": record.get("page"),
                "chunk_index": len(chunks),
                "text": part,
            })

    return chunks
