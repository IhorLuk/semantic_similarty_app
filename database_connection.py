from sqlalchemy import create_engine, insert, text, select, desc, Column, Integer, String, Text, VARCHAR, Table, MetaData
from sqlalchemy.orm import declarative_base, mapped_column, Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

# define base for openAI table
Base = declarative_base()

# table for embeddings
class DocumentOpenai(Base):
    __tablename__ = "test_vectors_openai"
    __table_args__ = {'schema':'test_vectors'}

    id = Column(VARCHAR, primary_key=True)
    embedding = Column(Vector(1536))

# table for text of the comments
class DocumentComments(Base):
    __tablename__ = "comments"
    __table_args__ = {'schema':'test_vectors'}

    id = Column(VARCHAR, primary_key=True)
    comment_text = Column(Text)

# define session - it is the main class to query table
class Session():
    def __init__(self, CONNSTR):
        self.CONNSTR = CONNSTR
        
    def create_engine(self):
        
        self.engine = create_engine(self.CONNSTR)
        
    def connect_session(self):
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def calculate_neighbors(self, DatabaseDocument, embed_query, k=5):
        self.neighbors = self.session.scalars(select(DatabaseDocument).order_by(DatabaseDocument.embedding.cosine_distance(embed_query)).limit(k))
        
        
    