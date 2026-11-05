import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Index

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
metadata = MetaData()

telemetry = Table(
    "telemetry", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("robot_id", String(32), nullable=False),
    Column("timestamp", Float, nullable=False),
    Column("x", Float), Column("y", Float),
    Column("speed", Float), Column("battery", Float),
    Column("motor_temp", Float), Column("vibration", Float),
    Column("motor_current", Float), Column("obstacle_distance", Float),
)
Index("ix_telemetry_robot_ts", telemetry.c.robot_id, telemetry.c.timestamp)

def init_db():
    metadata.create_all(engine)