from collections import namedtuple
from io import BytesIO
import docx
from docx.shared import Inches
import numpy as np
import os

from . import constants, datasets, models


# - add documentation
# - update changelog
# - improve summary table


StyleGuide = namedtuple('StyleGuide', [
    'table',
    'tbl_header',
    'tbl_body',
    'outfile',
    'header_level'
])


class Reporter:

    def __init__(self, template=None, styles=None):
        """
        Create a new Microsoft Word document (.docx) reporter object.


        Parameters
        ----------
        template : str, docx.Document instance, or None.
            If str, the path the word template to be used.
            If a docx.Document object, this object is used.
            If None, the default template is used.

        styles : Instance of StyleGuide, or None.
            Determines which styles to apply for each component in document.
            If None, then the default StyleGuide is used. Note that if a custom
            template is specified, a StyleGuide should generally also be
            specified.

        Returns
        -------
        None.
        """

        if template is None:
            template = os.path.join(
                os.path.dirname(__file__),
                'templates/base.docx'
            )

        if styles is None:
            styles = StyleGuide('bmdsTbl', 'bmdsTblHeader', 'bmdsTblBody',
                                'bmdsOutputFile', 1)

        self.styles = styles
        self.doc = docx.Document(template)

    def add_session(self, session, title=None,
                    input_dataset=True, summary_table=True,
                    recommended_model=True, all_models=False):
        """
        Add an existing session to a Word report.


        Parameters
        ----------
        session : bmds.Session
            BMDS session to be included in reporting
        title : str or None
            Title to be used to refer to the session. If one is not provided, a
            default value is used.
        input_dataset : bool
            Include input dataset data table
        summary_table : bool
            Include model summary table
        recommended_model : bool
            Include the recommended model output and dose-response plot, if
            one exists
        all_models : bool
            Include all models output and dose-response plots

        Returns
        -------
        None.
        """

        if title is None:
            title = 'BMDS output results'

        self.doc.add_heading(title, self.styles.header_level)

        if input_dataset:
            self._add_dataset(session.dataset)

        if summary_table:
            self._add_session_summary_table(session)

        if recommended_model and all_models:
            self._add_recommended_model(session)
            self._add_all_models(session, except_recommended=True)
        elif recommended_model:
            self._add_recommended_model(session)
        elif all_models:
            self._add_all_models(session, except_recommended=False)

    def save(self, filename):
        """
        Save document to a file.


        Parameters
        ----------
        filename : str
            The output string filename

        """
        self.doc.save(os.path.expanduser(filename))

    def _add_dataset(self, dataset):

        self.doc.add_heading('Input dataset', self.styles.header_level + 1)
        hdr = self.styles.tbl_header

        if isinstance(dataset, datasets.DichotomousDataset):

            tbl = self.doc.add_table(dataset.num_dose_groups + 1, 3,
                                     style=self.styles.table)

            self._write_cell(tbl.cell(0, 0), 'Dose', style=hdr)
            self._write_cell(tbl.cell(0, 1), 'Incidence', style=hdr)
            self._write_cell(tbl.cell(0, 2), 'Number of animals', style=hdr)
            for i, vals in enumerate(zip(dataset.doses,
                                         dataset.incidences,
                                         dataset.ns)):
                self._write_cell(tbl.cell(i + 1, 0), vals[0])
                self._write_cell(tbl.cell(i + 1, 1), vals[1])
                self._write_cell(tbl.cell(i + 1, 2), vals[2])

        elif isinstance(dataset, datasets.ContinuousDataset):

            tbl = self.doc.add_table(dataset.num_dose_groups + 1, 4,
                                     style=self.styles.table)

            self._write_cell(tbl.cell(0, 0), 'Dose', style=hdr)
            self._write_cell(tbl.cell(0, 1), 'Number of animals', style=hdr)
            self._write_cell(tbl.cell(0, 2), 'Mean', style=hdr)
            self._write_cell(tbl.cell(0, 3), 'Standard deviation', style=hdr)
            for i, vals in enumerate(zip(dataset.doses,
                                         dataset.ns,
                                         dataset.means,
                                         dataset.stdevs)):
                self._write_cell(tbl.cell(i + 1, 0), vals[0])
                self._write_cell(tbl.cell(i + 1, 1), vals[1])
                self._write_cell(tbl.cell(i + 1, 2), vals[2])
                self._write_cell(tbl.cell(i + 1, 3), vals[3])

        elif isinstance(dataset, datasets.ContinuousIndividualDataset):

            tbl = self.doc.add_table(dataset.dataset_length + 1, 2,
                                     style=self.styles.table)

            self._write_cell(tbl.cell(0, 0), 'Dose', style=hdr)
            self._write_cell(tbl.cell(0, 1), 'Response', style=hdr)
            for i, vals in enumerate(zip(dataset.individual_doses,
                                         dataset.responses)):
                self._write_cell(tbl.cell(i + 1, 0), vals[0])
                self._write_cell(tbl.cell(i + 1, 1), vals[1])

    def _write_cell(self, cell, value, style=None):

        if style is None:
            style = self.styles.tbl_body

        if isinstance(value, float):
            if np.isclose(value, 0.):
                tmpl = '{}'
            elif value < 0.001:
                tmpl = '{:.1E}'
            else:
                tmpl = '{:.3f}'
            value = tmpl.format(value)

        cell.paragraphs[0].text = str(value)
        cell.paragraphs[0].style = style

    def _add_session_summary_table(self, session):
        self.doc.add_heading('Summary table', self.styles.header_level + 1)
        hdr = self.styles.tbl_header

        if session.dtype in constants.CONTINUOUS_DTYPES:
            tbl = self.doc.add_table(len(session.models) + 2, 6,
                                     style=self.styles.table)

            # write headers
            self._write_cell(tbl.cell(0, 0), 'Model', style=hdr)
            self._write_cell(tbl.cell(0, 1), 'Variance p-value', style=hdr)
            self._write_cell(tbl.cell(0, 2), 'Goodness of fit', style=hdr)
            self._write_cell(tbl.cell(1, 2), 'p-value', style=hdr)
            self._write_cell(tbl.cell(1, 3), 'AIC', style=hdr)
            self._write_cell(tbl.cell(0, 4), 'BMD', style=hdr)
            self._write_cell(tbl.cell(0, 5), 'BMDL', style=hdr)

            # merge header columns
            tbl.cell(0, 0).merge(tbl.cell(1, 0))
            tbl.cell(0, 1).merge(tbl.cell(1, 1))
            tbl.cell(0, 2).merge(tbl.cell(0, 3))
            tbl.cell(0, 4).merge(tbl.cell(1, 4))
            tbl.cell(0, 5).merge(tbl.cell(1, 5))

        else:
            tbl = self.doc.add_table(len(session.models) + 2, 5,
                                     style=self.styles.table)

            # write headers
            self._write_cell(tbl.cell(0, 0), 'Model', style=hdr)
            self._write_cell(tbl.cell(0, 1), 'Goodness of fit', style=hdr)
            self._write_cell(tbl.cell(1, 1), 'p-value', style=hdr)
            self._write_cell(tbl.cell(1, 2), 'AIC', style=hdr)
            self._write_cell(tbl.cell(0, 3), 'BMD', style=hdr)
            self._write_cell(tbl.cell(0, 4), 'BMDL', style=hdr)

            # merge header columns
            tbl.cell(0, 0).merge(tbl.cell(1, 0))
            tbl.cell(0, 1).merge(tbl.cell(0, 2))
            tbl.cell(0, 3).merge(tbl.cell(1, 3))
            tbl.cell(0, 4).merge(tbl.cell(1, 4))

        # write body
        for i, model in enumerate(session.models):
            self._to_docx_summary_row(model, tbl, i + 2)

    def _to_docx_summary_row(self, model, tbl, idx):

        output = getattr(model, 'output', {})
        if isinstance(model, models.Dichotomous):

            self._write_cell(tbl.cell(idx, 0), model.name)
            self._write_cell(tbl.cell(idx, 1), output.get('p_value4', '-'))
            self._write_cell(tbl.cell(idx, 2), output.get('AIC', '-'))
            self._write_cell(tbl.cell(idx, 3), output.get('BMD', '-'))
            self._write_cell(tbl.cell(idx, 4), output.get('BMDL', '-'))

        elif isinstance(model, models.Continuous):

            self._write_cell(tbl.cell(idx, 0), model.name)
            self._write_cell(tbl.cell(idx, 1), output.get('p_value2', '-'))
            self._write_cell(tbl.cell(idx, 2), output.get('p_value4', '-'))
            self._write_cell(tbl.cell(idx, 3), output.get('AIC', '-'))
            self._write_cell(tbl.cell(idx, 4), output.get('BMD', '-'))
            self._write_cell(tbl.cell(idx, 5), output.get('BMDL', '-'))

    def _model_to_docx(self, model):
        if model.has_successfully_executed:
            # print figure
            fig = model.plot()
            with BytesIO() as f:
                fig.savefig(f)
                self.doc.add_picture(f, width=Inches(6))
            fig.clf()

            # print output file
            self.doc.add_paragraph(model.outfile, style=self.styles.outfile)
        else:
            self.doc.add_paragraph('No .OUT file was created.')

    def _add_recommended_model(self, session):
        self.doc.add_heading('Recommended model', self.styles.header_level + 1)
        if hasattr(session, 'recommended_model') and \
                session.recommended_model is not None:
            self._model_to_docx(session.recommended_model)
        else:
            p = self.doc.add_paragraph()
            p.add_run('No model was recommended as a best-fitting model.')\
             .italic = True

    def _add_all_models(self, session, except_recommended=False):
        if except_recommended:
            self.doc.add_heading('All other models',
                                 self.styles.header_level + 1)
        else:
            self.doc.add_heading('All model outputs',
                                 self.styles.header_level + 1)

        for model in session.models:

            if model.recommended and except_recommended:
                continue

            self._model_to_docx(model)
