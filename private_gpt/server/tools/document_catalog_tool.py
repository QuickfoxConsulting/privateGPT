import re
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from private_gpt.users import crud, models


@dataclass(frozen=True)
class DocumentCatalogResult:
    answer: str


class DocumentCatalogTool:
    """Deterministic document inventory answers backed by the documents table."""

    _DOC_WORD = r"(doc|docs|document|documents|doccument|doccuments|file|files)"
    _COUNT_PATTERNS = (
        rf"\bhow many\b.*\b{_DOC_WORD}\b",
        rf"\b(number|count|total)\b.*\b{_DOC_WORD}\b",
        rf"\b{_DOC_WORD}\b.*\b(number|count|total)\b",
    )
    _LIST_PATTERNS = (
        rf"\b(list|show|display|name|names)\b.*\b{_DOC_WORD}\b",
        rf"\bwhat\b.*\b{_DOC_WORD}\b.*\b(have|available|access)\b",
        rf"\bwhich\b.*\b{_DOC_WORD}\b.*\b(have|available|access)\b",
    )

    @classmethod
    def can_answer(cls, question: str) -> bool:
        normalized = cls._normalize(question)
        if not normalized:
            return False

        patterns = cls._COUNT_PATTERNS + cls._LIST_PATTERNS
        return any(re.search(pattern, normalized) for pattern in patterns)

    def answer(
        self, *, db: Session, user_id: int, question: str
    ) -> DocumentCatalogResult | None:
        if not self.can_answer(question):
            return None

        user = crud.user.get(db, id=user_id)
        if not user:
            return None

        documents = self._get_accessible_documents(db, user)
        wants_list = self._wants_list(question)
        total_count = len(documents)
        enabled_count = sum(1 for doc in documents if doc.is_enabled)

        if wants_list:
            answer = self._format_list_answer(documents, total_count, enabled_count)
        else:
            answer = self._format_count_answer(total_count, enabled_count)

        return DocumentCatalogResult(answer=answer)

    def _get_accessible_documents(
        self, db: Session, user: models.User
    ) -> Sequence[models.Document]:
        role = (
            user.user_role.role.name if user.user_role and user.user_role.role else None
        )

        if role in {"SUPER_ADMIN", "OPERATOR"}:
            return crud.documents.get_multi_documents(db).all()

        return crud.documents.get_documents_by_departments(
            db, department_id=user.department_id
        ).all()

    @classmethod
    def _wants_list(cls, question: str) -> bool:
        normalized = cls._normalize(question)
        return any(re.search(pattern, normalized) for pattern in cls._LIST_PATTERNS)

    @staticmethod
    def _normalize(question: str) -> str:
        return " ".join((question or "").lower().strip().split())

    @staticmethod
    def _format_count_answer(total_count: int, enabled_count: int) -> str:
        if total_count == enabled_count:
            return f"You have access to {total_count} document{'s' if total_count != 1 else ''}."

        disabled_count = total_count - enabled_count
        return (
            f"You have access to {total_count} document{'s' if total_count != 1 else ''}: "
            f"{enabled_count} enabled and {disabled_count} disabled."
        )

    @staticmethod
    def _format_list_answer(
        documents: Sequence[models.Document], total_count: int, enabled_count: int
    ) -> str:
        if not documents:
            return "You do not currently have access to any documents."

        visible_documents = list(documents[:20])
        names = "\n".join(f"- {doc.filename}" for doc in visible_documents)
        suffix = ""
        if total_count > len(visible_documents):
            suffix = f"\n\nShowing first {len(visible_documents)} of {total_count} documents."

        status_note = ""
        if total_count != enabled_count:
            status_note = f" ({enabled_count} enabled)"

        return f"You have access to {total_count} document{'s' if total_count != 1 else ''}{status_note}:\n\n{names}{suffix}"
