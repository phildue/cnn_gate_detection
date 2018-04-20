import os

from utils.workdir import cd_work

cd_work()

os.system('./lib/tf-models/research/object/train.py '
          '--logtostderr '
          '--train_dir=logs/ssd_industrial/ '
          '--pipeline_config_path=logs/ssd_industrial/pipeline.config')
