# set up steps for server
1. `cd Grounded-SAM-2`
2. `conda create --name SAM-Endpoint python=3.10`
3. `conda activate SAM-Endpoint`
4. `pip install -r requirements.txt`
5. `uvicorn grounded_sam2_api:app --reload`


## take a look at the 
- take a look at `sam-request.py`