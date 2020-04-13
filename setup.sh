PYTHONPATH=/home/david/dev/HA-Tracking/HATracking/libs/SiamMask:/home/david/dev/HA-Tracking/HATracking/libs/SiamMask/experiments/siammask_sharp:$PYTHONPATH
$PYTHONPATH

python tools/demo.py --resume experiments/siammask_sharp/SiamMask_DAVIS.pth --config experiments/siammask_sharp/config_davis.json --base_path data/tennis
