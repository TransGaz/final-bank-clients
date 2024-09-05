from typing import List, Optional
from db_clients import *
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from fastapi import Depends, FastAPI, Body
from fastapi.responses import JSONResponse
import  uvicorn
import psycopg2
from config import connection_string
from sqlalchemy import URL, create_engine




engine = create_engine(connection_string)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

db = SessionLocal()



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_model=dict)
async def root():
    """Возвращает строку с приветствием банка"""
    return {"message":'Добро пожаловать в банк!'}


@app.get("/api/clients", summary="Получить список клиентов!" )
def read_clients(db: Session = Depends(get_db)):
    """Функция, возвращает список клиентов банка на основе которых идет исследование"""
    return db.query(Client).all()


@app.get("/api/client/{id}", summary="Получить клиента по id" )
def read_client(id, db: Session = Depends(get_db)):
    """Функция, которая возвращает клиента банка по id"""

    # получаем клиента по id
    client = db.query(Client).filter(Client.id == id).first()
    # если не найден, отправляем статусный код и сообщение об ошибке
    if client is None:
        return JSONResponse(status_code=404, content={"message": "Такоого клиента у нас нет!"})
    return client

# g