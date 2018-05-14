import argparse

from modelzoo.backend.tensor.ModelPlot import ModelPlot
from modelzoo.models.ModelBuilder import ModelBuilder
from utils.workdir import cd_work

cd_work()
parser = argparse.ArgumentParser()
parser.add_argument("name", help="model name",
                    type=str)

parser.add_argument("output_file", help="Output File", type=str)

args = parser.parse_args()
name = args.name
output_file = args.output_file

model = ModelBuilder.get_model(name)

ModelPlot(model).save(output_file)
