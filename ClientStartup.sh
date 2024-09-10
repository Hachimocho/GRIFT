ssh -i $1 -f -N brg2890@nsf-gpu.main.ad.rit.edu -L 9998:nsf-gpu.main.ad.rit.edu:9998 &
conda activate DeepEARL &
python3 AgentClient.py