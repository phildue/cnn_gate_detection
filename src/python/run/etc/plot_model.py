import argparse

from modelzoo.backend.tensor.ModelPlot import ModelPlot
from modelzoo.models.ModelFactory import ModelFactory
from utils.workdir import cd_work

cd_work()
parser = argparse.ArgumentParser()
parser.add_argument("name", help="model name",
                    type=str)

parser.add_argument("output_file", help="Output File", type=str)

args = parser.parse_args()
name = args.name
output_file = args.output_file

model = ModelFactory.build(name)

ModelPlot(model).save(output_file)
