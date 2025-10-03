# FastAPI 15min tutorial: https://www.youtube.com/watch?v=iWS9ogMPOI0&t=96s

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str = None
    is_done: bool = False

items = []

@app.get('/')
def root():
    return {'Hello': 'World'}

@app.post('/items')
def create_item(item: Item):
    items.append(item)
    return items

@app.get('/items', response_model=list[Item])
def list_items(limit: int=10):
    return items[0:limit]

@app.get('/items/{item_id}', response_model=Item)
def get_item(item_id: int) -> Item: # Note - FastAPI will automatically convert to int here
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f'Item {item_id} not found.')

# How to get to automatic, interactive documentation page:
# http://127.0.0.1:8000/docs#/

# Using curl: -X for request, -H for header, -d for data

# In terminal where main.py is located > uvicorn main:app --reload
# ^ Replace "main" with fname and "app" with the FastAPI() instance name

# In new terminal (before changing the post data type from "str" to "Item")
# curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8000/items?item=apple
# curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8000/items?item=banana
# ^ Note: The header may not be necessary?

# curl -X GET http://127.0.0.1:8000/items/0
# curl -X GET http://127.0.0.1:8000/items/1
# curl -X GET http://127.0.0.1:8000/items/2 # This would yield an "Internal Server Error" msg

# curl -X GET http://127.0.0.1:8000/items?limit=3
# curl -X GET http://127.0.0.1:8000/items? # This would default to 10 (defined in function header)

# After changing the post data type from "str" to "Item", use a json payload now
# curl -X POST -H "Content-Type: application/json" -d "{\"text\" :\"apple\"}" http://127.0.0.1:8000/items
# ^ Note that on Windows, a \ is needed to close inner quotes (can't use single quotes)
# ^ Also note that apparently the header was needed for this command