class TableFootnote:
    def __init__(self):
        self.footnotes: dict[str, str] = {}

    def __len__(self):
        return len(self.footnotes)

    def _add_footnote_character(self, p, symbol):
        run = p.add_run(symbol)
        run.font.superscript = True

    def add_footnote(self, p, text: str):
        """Append a superscript footnote icon to the text"""
        if text not in self.footnotes:
            icon = chr(97 + len(self.footnotes))
            self.footnotes[text] = icon
        self._add_footnote_character(p, self.footnotes[text])

    def add_footnote_text(self, doc, style: str):
        """Print all footnotes for this instance"""
        for text, char in self.footnotes.items():
            p = doc.add_paragraph("", style=style)
            self._add_footnote_character(p, char)
            p.add_run(f" {text}")
