#!/usr/bin/env python3
"""
OpenRecall Database Initialization Script
This script creates the necessary database tables for OpenRecall
"""

import sqlite3
import os
from pathlib import Path

# OpenRecall data directory
data_dir = Path(os.path.expandvars(r"C:\Users\Irak\AppData\Roaming\openrecall"))
db_path = data_dir / "recall.db"

print(f"Initializing OpenRecall database at: {db_path}")

# Ensure the directory exists
data_dir.mkdir(parents=True, exist_ok=True)

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the entries table
print("Creating 'entries' table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app TEXT,
        title TEXT,
        text TEXT,
        timestamp INTEGER,
        embedding BLOB
    )
""")

# Create indexes for better query performance
print("Creating indexes...")
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_timestamp ON entries(timestamp)
""")

cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_app ON entries(app)
""")

# Commit changes
conn.commit()

# Verify tables were created
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("\nDatabase tables created:")
for table in tables:
    print(f"  - {table[0]}")

# Check if entries table exists and show its schema
cursor.execute("PRAGMA table_info(entries)")
columns = cursor.fetchall()
print("\nEntries table schema:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

conn.close()

print("\n[OK] Database initialized successfully!")
print(f"Database location: {db_path}")
print("\nYou can now start OpenRecall with: python run_openrecall.py")
