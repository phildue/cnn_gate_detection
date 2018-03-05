class Report:
    def __init__(self, template_path='dvlab/resource/report_template.tex'):
        self.template_path = template_path
        self.modelname = None
        self.notes = None
        self.params = None
        self.model_plot = False

    def save(self, filename):
        with open(self.template_path, 'r') as f:
            content = f.read()

        if self.modelname is not None:
            content = content.replace('\#modelname', self.modelname)
        if self.notes is not None:
            content = content.replace('\#notes', self.notes)
        if self.params is not None:
            for key, value in self.params:
                content = content.replace('\#' + key, value)

        if self.model_plot:
            content = content.replace('%1', '')

        with open(filename, 'w') as f:
            f.write(content)

        print('Report written to ' + filename)
