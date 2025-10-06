import asyncio
import pickle 
from fastapi import FastAPI , HTTPException , Depends 
from pydantic import BaseModel , Field
from contextlib import asynccontextmanager
from fastapi.security import APIKeyHeader
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd


loyalCustmer_model = None # pickle.load("loyalCustmer_LogisticRegression1.pkl")
custmer_order_returns = None
csutomer_CLV = None 
kmeans = None


def load_ecommerce_models():
    global loyalCustmer_model
    global custmer_order_returns
    global csutomer_CLV
    global kmeans
    
    with open("./ecommerce_large/loyalCustmer_LogisticRegression1.pkl", "rb") as f:
        loyalCustmer_model = pickle.load(f)
    with open("./ecommerce_large/custmerReturns_XGBClassifier.pkl", "rb") as f:
        custmer_order_returns = pickle.load(f)
    with open("./ecommerce_large/CsutomerCLV_XGBRegressor.pkl", "rb") as f:
        csutomer_CLV = pickle.load(f)
    with open("./ecommerce_large/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

@asynccontextmanager
async def lifespan(app : FastAPI ):
    load_ecommerce_models()
    print("Models Loaded Sussfully ... ")

    yield

    print("Models Closed Sussfully ... ")

    # -------- RateLimiter --------
class RateLimiter:
    def __init__(self, limit: int = 10, window_seconds: int = 10):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)  # {api_key: [timestamps]}

    def is_rate_limited(self, api_key: str) -> bool:
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)

        # نشيل الطلبات القديمة
        self.requests[api_key] = [
            ts for ts in self.requests[api_key] if ts > window_start
        ]

        if len(self.requests[api_key]) >= self.limit:
            return True

        self.requests[api_key].append(now)
        return False


# -------- App + Security --------

app = FastAPI(lifespan=lifespan)
rate_limiter = RateLimiter(limit=5, window_seconds=10)
header_schema = APIKeyHeader(name="X-API-Key", auto_error=True)
API_SECRET_KEY = "12345678"


class Loyal_Customer(BaseModel):
    AOV : int  = Field(...)
    CLV : int  = Field(...)
    Recency : int  = Field(...)
    return_rate : int  = Field(...)
    partial_return_rate : int  = Field(...)
    Canceled : int  = Field(...)
    Placed : int  = Field(...)
    cancel_rate : int  = Field(...)
    avg_spend : int  = Field(...)
    engagement_score : int  = Field(...)

    gender : str = Field(... , min_length=4 , max_length=50)
    country : str = Field(... , min_length=5 , max_length=50)


def predict_customers_loyality(data : Loyal_Customer , api_key : str) :
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="API Key not correct ❌")
    
    if rate_limiter.is_rate_limited(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests ⛔")
    
    features = pd.DataFrame([{
        "AOV": data.AOV,
        "CLV": data.CLV,
        "Recency": data.Recency,
        "return_rate": data.return_rate,
        "partial_return_rate": data.partial_return_rate,
        "Canceled": data.Canceled,
        "Placed": data.Placed,
        "cancel_rate": data.cancel_rate,
        "avg_spend": data.avg_spend,
        "engagement_score": data.engagement_score,
        "gender": data.gender,
        "country": data.country
    }])


    predicated = loyalCustmer_model.predict(features)[0]
    probability = loyalCustmer_model.predict_proba(features)[0]

    return {
        "is_loyal": int(predicated),
        "probabilities": {
            "not_loyal (class 0)": float(probability[0]),
            "loyal (class 1)": float(probability[1])
        }
    }
@app.get("/health")
async def health_check():
    if loyalCustmer_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded ❌")
    return {"status": "OK ✅", "message": "Model is loaded and API is healthy"}

@app.post("/predict-cutomer-loyality")
async def predict_cutomer_loyality(data : Loyal_Customer , api_key : str = Depends(header_schema)):

    restult = await asyncio.to_thread(predict_customers_loyality , data , api_key )
    return restult

"""
test data : 
    {
    "AOV": 250,
    "CLV": 1200,
    "Recency": 15,
    "return_rate": 3,
    "partial_return_rate": 1,
    "Canceled": 0,
    "Placed": 10,
    "cancel_rate": 5,
    "avg_spend": 800,
    "engagement_score": 75,
    "gender": "male",
    "country": "egypt"
    }

"""


# ----------------- custmerReturns_XGBClassifier ------------------------

# numeric_features = [ 'is_loyal', 'AOV', 'CLV', 'Frequency', 'Monetary',
#        'Recency', 'Canceled',
#          'Placed','cancel_rate', 'avg_spend', 'engagement_score'
#     ]

# categorical_features = ['gender','country']

class CustmerReturns(BaseModel):
    AOV : int  = Field(...)
    CLV : int  = Field(...)
    Recency : int  = Field(...)
    is_loyal : int  = Field(...)
    Monetary : int  = Field(...)
    Canceled : int  = Field(...)
    Placed : int  = Field(...)
    cancel_rate : int  = Field(...)
    avg_spend : int  = Field(...)
    engagement_score : int  = Field(...)
    Frequency : int  = Field(...)

    gender : str = Field(... , min_length=4 , max_length=50)
    country : str = Field(... , min_length=5 , max_length=50)


def check_order_will_be_return(data : CustmerReturns , api_key : str):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="API Key not correct ❌")
    
    if rate_limiter.is_rate_limited(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests ⛔")
    
    features = pd.DataFrame([{
        "AOV": data.AOV,
        "CLV": data.CLV,
        "Recency": data.Recency,
        "is_loyal": data.is_loyal,
        "Monetary": data.Monetary,
        "Canceled": data.Canceled,
        "Placed": data.Placed,
        "cancel_rate": data.cancel_rate,
        "avg_spend": data.avg_spend,
        "engagement_score": data.engagement_score,
        "Frequency": data.Frequency,
        "gender": data.gender,
        "country": data.country
    }])

    predicated = custmer_order_returns.predict(features)[0]
    probability = custmer_order_returns.predict_proba(features)[0]

    return {
        "will_return": int(predicated),
        "probabilities": {
            "will_return (class 0)": float(probability[0]),
            "will_not_return (class 1)": float(probability[1])
        }
    }

@app.post("/custmer-order-return")
async def custmer_order_return(data : CustmerReturns , api_key : str = Depends(header_schema)):

    restult = await asyncio.to_thread(check_order_will_be_return , data , api_key )
    return restult

"""
test data : 
    {
    "AOV": 134,
    "CLV": 403,
    "Recency": 398,
    "is_loyal": 1,
    "Monetary": 403,
    "Canceled": 0,
    "Placed": 67,
    "cancel_rate": 0,
    "avg_spend": 406,
    "engagement_score": 50,
    "Frequency": 3,
    "gender": "Male",
    "country": "France"
    }



"""


# ----------------- CsutomerCLV_XGBRegressor ------------------------


# numeric_features = [ 'is_loyal', 
#        'Recency', 'Canceled', 'avg_spend' ,
#          'Placed','Recency'
#     ]

# categorical_features = ['gender','country']

class Custmer_CLV(BaseModel):
    Recency : int  = Field(...)
    is_loyal : int  = Field(...)
    Canceled : int  = Field(...)
    Placed : int  = Field(...)
    avg_spend : int  = Field(...)

    gender : str = Field(... , min_length=4 , max_length=50)
    country : str = Field(... , min_length=5 , max_length=50)


def check_order_will_be_return(data : Custmer_CLV , api_key : str):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="API Key not correct ❌")
    
    if rate_limiter.is_rate_limited(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests ⛔")
    
    features = pd.DataFrame([{
        "Recency": data.Recency,
        "is_loyal": data.is_loyal,
        "Canceled": data.Canceled,
        "Placed": data.Placed,
        "avg_spend": data.avg_spend,
        "gender": data.gender,
        "country": data.country
    }])

    predicated = csutomer_CLV.predict(features)[0]

    return {
        "Customer CLV": int(predicated),
    }

@app.post("/custmer-clv")
async def custmer_clv_predict(data : Custmer_CLV , api_key : str = Depends(header_schema)):

    restult = await asyncio.to_thread(check_order_will_be_return , data , api_key )
    return restult

"""
test data : 
    {
    "Recency": 398,
    "is_loyal": 1,
    "Canceled": 0,
    "Placed": 67,
    "avg_spend": 406,
    "gender": "Male",
    "country": "France"
    }


"""


# ----------------------- Customer Segmenation -----------------------------

# ['CLV', 'Frequency', 'Recency', 'AOV', 'return_rate']

class Custmer_Segmentation(BaseModel):
    Recency : int  = Field(...)
    Frequency : int  = Field(...)
    CLV : int  = Field(...)
    AOV : int  = Field(...)
    return_rate : int  = Field(...)
# features = ['CLV', 'Frequency', 'Recency', 'AOV', 'return_rate']


def predict_customer_segment(data : Custmer_Segmentation , api_key : str):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="API Key not correct ❌")
    
    if rate_limiter.is_rate_limited(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests ⛔")
    
    features = pd.DataFrame([{
        "CLV": data.CLV,
        "Frequency": data.Frequency,
        "Recency": data.Recency,
        "AOV": data.AOV,
        "return_rate": data.return_rate
    }])

    predicated = kmeans.predict(features)[0]

    return {
        "Customer Segement": int(predicated),
    }

@app.post("/custmer-segement")
async def predict_custmer_segement(data : Custmer_Segmentation , api_key : str = Depends(header_schema)):

    restult = await asyncio.to_thread(predict_customer_segment , data , api_key )
    return restult

"""
test data : 
    {
    "CLV": 0,
    "Frequency": 1,
    "Recency": 398,
    "AOV": 67,
    "return_rate": 406
    }


"""