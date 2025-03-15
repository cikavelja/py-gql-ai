from fastapi import FastAPI, Depends, HTTPException
from strawberry.fastapi import GraphQLRouter
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import Column, Integer, String, Text, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import strawberry
import jwt
import bcrypt
from typing import List, Optional
from fastapi.security import OAuth2PasswordBearer
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import json

# Load Local Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & lightweight model

# Database Configuration
DATABASE_URL = "postgresql+asyncpg://admin:admin123@127.0.0.1:5432/gqlpy"
engine = create_async_engine(
    DATABASE_URL, 
    echo=True,
    pool_size=20,  # Increase the pool size
    max_overflow=40,  # Allow more connections when needed
    pool_timeout=60  # Increase timeout before giving up on connections
)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

# Authentication Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User Model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="user")

# Article Model
class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    vector = Column(Text)  # Store vector embedding as JSON
    search_text = Column(Text)  # Full-text search field

# Async Database Dependency
@asynccontextmanager
async def get_db_session():
    async with SessionLocal() as session:
        yield session

# Utility Functions
async def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

async def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

async def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        async with get_db_session() as db:
            result = await db.execute(select(User).filter(User.username == payload.get("sub")))
            user = result.scalar_one_or_none()
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# GraphQL Schema
@strawberry.type
class UserType:
    id: int
    username: str
    role: str

@strawberry.type
class ArticleType:
    id: int
    title: str
    content: str

@strawberry.type
class Query:
    @strawberry.field
    async def list_articles(self, info) -> List[ArticleType]:
        db: AsyncSession = info.context["db"]
        result = await db.execute(select(Article))
        return result.scalars().all()

    @strawberry.field
    async def get_article(self, info, id: int) -> Optional[ArticleType]:
        db: AsyncSession = info.context["db"]
        result = await db.execute(select(Article).filter(Article.id == id))
        return result.scalar_one_or_none()
    
@strawberry.field
async def search_articles_semantic(self, info, query: str) -> List[ArticleType]:
    db: AsyncSession = info.context["db"]
    
    # Convert user query into a vector
    query_vector = embedding_model.encode(query).tolist()
    query_vector_str = f"ARRAY{query_vector}"  

    # Query the database using cosine similarity
    result = await db.execute(
        f"""
        SELECT id, title, content, 1 - (vector <=> {query_vector_str}) AS similarity_score
        FROM articles
        ORDER BY similarity_score DESC
        LIMIT 5;
        """
    )
    
    articles = []
    for row in result.fetchall():
        articles.append({
            "id": row.id,
            "title": row.title,
            "content": row.content,
            "similarity_score": round(row.similarity_score, 4)  
        })
    
    return articles


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_article(self, info, title: str, content: str) -> ArticleType:
        db: AsyncSession = info.context["db"]

        # Generate embeddings using the local model
        embedding_vector = embedding_model.encode(content).tolist()  
        
        article = Article(title=title, content=content, vector=json.dumps(embedding_vector))
        db.add(article)
        await db.commit()
        await db.refresh(article)
        return article

    @strawberry.mutation
    async def register_user(self, info, username: str, password: str) -> UserType:
        db: AsyncSession = info.context["db"]
        user = User(username=username, password_hash=await hash_password(password))
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

async def get_context():
    async with get_db_session() as db:
        return {"db": db}

schema = strawberry.Schema(query=Query, mutation=Mutation)

# FastAPI App
app = FastAPI(title="AI-Powered GraphQL API", description="GraphQL API with Full-Text & Vector Search")

gql_router = GraphQLRouter(schema, context_getter=get_context)
app.include_router(gql_router, prefix="/graphql")

@app.post("/token")
async def login(form_data: dict):
    async with get_db_session() as db:
        result = await db.execute(select(User).filter(User.username == form_data["username"]))
        user = result.scalar_one_or_none()
        if not user or not await verify_password(form_data["password"], user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"access_token": await create_access_token({"sub": user.username}), "token_type": "bearer"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    return {"message": "You have access", "user": user}
