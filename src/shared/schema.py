from pydantic import BaseModel


__all__ = ["Place"]


class Place(BaseModel):
    # Python row contract for the Postgres places table.
    id: str
    place_description: str
    summary: str

    @classmethod
    def db_columns(cls) -> list[str]:
        """Return columns in the same order as the Pydantic model fields."""
        return list(cls.model_fields)

    def db_values(self) -> tuple:
        return tuple(getattr(self, column) for column in self.db_columns())

    @property
    def embedding_text(self) -> str:
        return f"{self.summary}\n{self.place_description}"

    @property
    def context_text(self) -> str:
        return f"id: {self.id}\nsummary: {self.summary}\ndescription: {self.place_description}"
