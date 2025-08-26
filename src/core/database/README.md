# Database Package

This package contains database schemas, initialization scripts, and database utilities for the USDCOP Trading RL System.

## 📁 Structure

```
src/core/database/
├── __init__.py              # Package initialization
├── database_manager.py      # Database management utilities
├── init-databases.sql       # Database initialization SQL
├── trading-schema.sql       # Trading system schema
└── README.md                # This file
```

## 🗄️ Database Files

### `init-databases.sql`
- Database initialization script
- Creates database structure
- Sets up initial tables and indexes

### `trading-schema.sql`
- Trading system specific schema
- Tables for orders, trades, positions
- Performance metrics and analytics

## 🚀 Usage

### Initialize Database

```bash
# Using Make command
make init-db

# Using Python script directly
python scripts/init_database.py
```

### Check Database Status

```bash
# Check database tables and status
make db-status
```

### Python API

```python
from src.core.database.database_manager import DatabaseManager

# Initialize database
with DatabaseManager() as db:
    success = db.initialize_database()
    
    if success:
        # Execute queries
        results = db.execute_query("SELECT * FROM trades")
        print(f"Found {len(results)} trades")
```

## 🔧 Database Manager

The `DatabaseManager` class provides:

- **Database Initialization**: Creates database and applies schemas
- **Connection Management**: Handles SQLite connections
- **Query Execution**: Safe query execution with error handling
- **Context Manager**: Automatic connection cleanup

### Key Methods

- `initialize_database()`: Set up database with schemas
- `get_connection()`: Get database connection
- `execute_query(query, params)`: Execute SQL queries
- `close()`: Close database connection

## 📊 Database Schema

The system uses SQLite for simplicity and portability. Key tables include:

- **trades**: Trade execution records
- **orders**: Order management
- **positions**: Current positions
- **performance**: Performance metrics
- **data_quality**: Data quality tracking

## 🛠️ Development

### Adding New Tables

1. Add table definition to `trading-schema.sql`
2. Update `DatabaseManager` if needed
3. Test with `make init-db`

### Database Migrations

Use the migration system for schema changes:

```bash
make migrate-db
```

## 🔍 Troubleshooting

### Common Issues

1. **Database locked**: Check if another process is using the database
2. **Schema errors**: Verify SQL syntax in schema files
3. **Permission errors**: Ensure write access to data directory

### Debug Commands

```bash
# Check database file
ls -la data/trading.db

# Verify schema
sqlite3 data/trading.db ".schema"

# Check tables
sqlite3 data/trading.db ".tables"
```

## 📚 Related Documentation

- [Main README](../../../../README.md)
- [Architecture Guide](../../../../docs/ARCHITECTURE.md)
- [Configuration Guide](../../../../configs/)
