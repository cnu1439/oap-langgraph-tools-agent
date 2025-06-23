import sqlite3
import pandas as pd


class DatabaseInitializer:
    """Initialize database with CSV data."""

    def __init__(self, db_path: str = "hr_database.db"):
        self.db_path = db_path

    def create_database_from_csv(self, csv_file_path: str, table_name: str = "employees") -> str:
        """
        Create database and load CSV data into specified table.

        Args:
            csv_file_path: Path to the CSV file
            table_name: Name of the table to create

        Returns:
            Success message with database info
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)

            # Clean column names (remove spaces, special characters)
            df.columns = (
                df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '_')
            )

            # Create SQLite connection
            conn = sqlite3.connect(self.db_path)

            # Load data into database
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Get table info
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()

            conn.close()

            # Format column information
            columns_str = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

            return f"""Database '{self.db_path}' created successfully!
Table: {table_name}
Columns: {columns_str}
Records loaded: {len(df)}"""

        except Exception as e:
            return f"Error creating database: {str(e)}"
