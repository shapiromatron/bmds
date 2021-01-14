class TableFootnote(dict):
    def __init__(self):
        super().__init__()
        self.ascii_char = 96

    def add_footnote(self, p, text):
        if text not in self:
            self.ascii_char += 1
            self[text] = chr(self.ascii_char)
        self._add_footnote_character(p, self[text])

    def _add_footnote_character(self, p, symbol):
        run = p.add_run(symbol)
        run.font.superscript = True

    def add_footnote_text(self, doc, style):
        for text, char in self.items():
            p = doc.add_paragraph("", style=style)
            self._add_footnote_character(p, char)
            p.add_run(" {}".format(text))
