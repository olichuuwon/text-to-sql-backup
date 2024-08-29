import os
import re
import streamlit as st
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from prettytable import PrettyTable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama

# Set the Streamlit page configuration
st.set_page_config(page_title="LLM Tools", page_icon=":speech_balloon:")

# Load model configurations from environment variables
MODEL_NAME = os.getenv('MODEL_NAME', 'llama3:instruct')
MODEL_BASE_URL = os.getenv('MODEL_BASE_URL', 'http://model:11434')

# Sidebar for mode selection
st.sidebar.title("LLM Tools")
page = st.sidebar.selectbox("Select Mode", ["Database Mode", "General Mode"])

# Sidebar inputs for database connection, only shown in Database Mode
if page == "Database Mode":
    st.sidebar.title("Database Connection Settings")
    db_host = st.sidebar.text_input("Host", value=os.getenv('DB_HOST', 'postgres'))
    db_port = st.sidebar.text_input("Port", value=os.getenv('DB_PORT', '5432'))
    db_user = st.sidebar.text_input("User", value=os.getenv('DB_USER', 'user'))
    db_password = st.sidebar.text_input("Password", value=os.getenv('DB_PASSWORD', 'pass'), type="password")
    db_name = st.sidebar.text_input("Database", value=os.getenv('DB_NAME', 'chinook'))

    # Store connection details in session state
    if "db_uri" not in st.session_state:
        st.session_state.db_uri = None
    if "db" not in st.session_state:
        st.session_state.db = None

    # Function to get database URI
    def uri_database(user: str, password: str, host: str, port: str, database: str) -> str:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Update session state based on user input
    st.session_state.db_uri = uri_database(db_user, db_password, db_host, db_port, db_name)

    # Button to manually connect to the database
    if st.sidebar.button("Connect"):
        try:
            engine = create_engine(st.session_state.db_uri)
            with engine.connect() as connection:
                st.sidebar.success("Successfully connected to the database!")
                st.session_state.db = SQLDatabase.from_uri(st.session_state.db_uri)  # Initialize database connection
        except Exception as e:
            st.sidebar.error(f"Failed to connect to database: {e}")

# Function to get table schema
def get_table_schema(engine, table_name: str) -> PrettyTable:
    inspector = inspect(engine)
    schema_table = PrettyTable()
    schema_table.field_names = ["Column Name", "Data Type"]
    columns = inspector.get_columns(table_name)
    for column in columns:
        schema_table.add_row([column['name'], column['type']])
    return schema_table

# Function to get sample data
def get_sample_data(engine, table_name: str, sample_size: int = 1) -> PrettyTable:
    query = f'SELECT * FROM "{table_name}" LIMIT {sample_size};'
    df = pd.read_sql(query, engine)
    sample_data_table = PrettyTable()
    if not df.empty:
        sample_data_table.field_names = df.columns.tolist()
        for _, row in df.iterrows():
            sample_data_table.add_row(row.tolist())
    return sample_data_table

# Function to display table information
def display_table_schema(db_uri: str, sample_size: int = 1) -> None:
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    with st.expander("View Database Schema"):
        for table_name in inspector.get_table_names():
            st.write(f"**Table: {table_name}**")
            st.write("Schema:")
            st.write(get_table_schema(engine, table_name))
            st.write("Sample Data:")
            st.write(get_sample_data(engine, table_name, sample_size))

# Manual SQL Command Execution in the sidebar
def display_sql_execution(db_uri: str) -> None:
    engine = create_engine(db_uri)
    with st.expander("Run Statements Manually"):
        st.caption("Name 10 tracks.")
        st.code('SELECT * FROM "Track" LIMIT 10;', language="sql", line_numbers=False)
        st.caption("List of the top 50 customers based on their total sales, ordered in descending order.")
        st.code('SELECT "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName", (SELECT SUM("Invoice"."Total") FROM "Invoice" WHERE "Invoice"."CustomerId" = "Customer"."CustomerId") AS TotalSales FROM "Customer" ORDER BY TotalSales DESC LIMIT 50;', language="sql", line_numbers=False)
        sql_query = st.text_area("Enter SQL query:", height=100, key="sql_query_sidebar")
        if st.button("Fetch", key="run_sql_sidebar"):
            # Consider only the first part of the query (up to the first semicolon)
            if is_safe_query(sql_query):
                try:
                    first_part = sql_query.split(';', 1)[0].strip()
                    with engine.connect() as connection:
                        result = connection.execute(text(first_part))  # Wrap query with text()
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                        st.write("Query Result:")
                        st.dataframe(df)  # Display the result as a dataframe
                except Exception as e:
                    st.error(f"Error executing SQL query: {e}")
            else:
                st.error("Only single SELECT statements are allowed. The use of harmful keywords such as UNION, DROP, INSERT, UPDATE, DELETE, CREATE, ALTER, TRUNCATE, EXEC, EXECUTE, and XP_CMDSHELL, as well as comments (e.g., --, #, / /) or mismatched quotes, is prohibited.")


# Function to get SQL query chain
def get_sql_chain(db: SQLDatabase):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a PostgresSQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Instructions:
    - Use UTC timing for any date or time calculations.
    - Convert epoch timestamps to UTC timing if necessary.
    - Prioritize the use of SELECT statements with explicit column names; use "*" only when all columns are required.
    - Use JOIN operations judiciously to ensure data integrity when multiple tables are referenced. Avoid UNION unless absolutely necessary.
    - Limit the result set to a maximum of 100,000 rows to prevent large data returns.
    - Always use explicit table names or aliases when referencing columns to prevent ambiguity, especially if the column name exists in multiple tables.
    - Avoid using constructs not supported by PostgreSQL, such as CROSS APPLY.
    - Handle null values and ensure to include conditions that maintain data integrity in joins and filters.
    - Ensure that aggregation functions (e.g., COUNT, SUM, AVG) are used with GROUP BY clauses when necessary.
    - The syntax for the random function in PostgreSQL is: random()

    Write only the SQL query and nothing else.
    Do not wrap the SQL query in any other text, not even backticks.
    Do not reply to the user, and only respond with SQL queries.

    For example:

    Question: Which 3 genres have the most tracks?
    SQL Query: SELECT g."Name" AS GenreName, COUNT(*) AS track_count FROM "Track" t JOIN "Genre" g ON t."GenreId" = g."GenreId" GROUP BY g."Name" ORDER BY track_count DESC LIMIT 3;

    Question: Name 10 playlists.
    SQL Query: SELECT "Name" FROM "Playlist" LIMIT 10;

    Question: What are the 5 most recent invoices?
    SQL Query: SELECT * FROM "Invoice" ORDER BY "InvoiceDate" DESC LIMIT 5;

    Question: List the names and titles of employees and their managers.
    SQL Query: SELECT e1."FirstName" AS EmployeeFirstName, e1."LastName" AS EmployeeLastName, e1."Title" AS EmployeeTitle, e2."FirstName" AS ManagerFirstName, e2."LastName" AS ManagerLastName, e2."Title" AS ManagerTitle FROM "Employee" e1 LEFT JOIN "Employee" e2 ON e1."ReportsTo" = e2."EmployeeId";

    Question: What is the average unit price of tracks by genre?
    SQL Query: SELECT g."Name" AS GenreName, AVG(t."UnitPrice") AS AverageUnitPrice FROM "Track" t JOIN "Genre" g ON t."GenreId" = g."GenreId" GROUP BY g."Name" ORDER BY AverageUnitPrice DESC;

    Question: How many albums does each artist have?
    SQL Query: SELECT a."ArtistId", a."Name" AS ArtistName, COUNT(al."AlbumId") AS AlbumCount FROM "Artist" a JOIN "Album" al ON a."ArtistId" = al."ArtistId" GROUP BY a."ArtistId", a."Name" ORDER BY AlbumCount DESC;

    Question: List all customers from Canada.
    SQL Query: SELECT "CustomerId", "FirstName", "LastName", "Email" FROM "Customer" WHERE "Country" = 'Canada';

    Question: What is the total sales for each customer?
    SQL Query: SELECT c."CustomerId", c."FirstName", c."LastName", SUM(i."Total") AS TotalSales FROM "Customer" c JOIN "Invoice" i ON c."CustomerId" = i."CustomerId" GROUP BY c."CustomerId", c."FirstName", c."LastName" ORDER BY TotalSales DESC;

    Question: List the names of tracks that are longer than 5 minutes.
    SQL Query: SELECT "Name" FROM "Track" WHERE "Milliseconds" > 300000;

    Question: List the titles of albums by AC/DC.
    SQL Query: SELECT al."Title" FROM "Album" al JOIN "Artist" a ON al."ArtistId" = a."ArtistId" WHERE a."Name" = 'AC/DC';

    Question: What are the 5 most recent invoices in UTC?
    SQL Query: SELECT * FROM "Invoice" ORDER BY "InvoiceDate" AT TIME ZONE 'UTC' DESC LIMIT 5;

    Question: How many tracks are longer than 5 minutes?
    SQL Query: SELECT COUNT(*) FROM "Track" WHERE "Milliseconds" > 300000;

    Question: Get the first 100,000 rows from the track details.
    SQL Query: SELECT * FROM "Track" LIMIT 100000;

    Question: Show the total sales for each customer without using joins.
    SQL Query: SELECT "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName", (SELECT SUM("Invoice"."Total") FROM "Invoice" WHERE "Invoice"."CustomerId" = "Customer"."CustomerId") AS TotalSales FROM "Customer" LIMIT 100000;

    Question: What is the total number of tracks in each media type?
    SQL Query: SELECT m."Name" AS MediaTypeName, COUNT(t."TrackId") AS TrackCount FROM "Track" t JOIN "MediaType" m ON t."MediaTypeId" = m."MediaTypeId" GROUP BY m."Name" ORDER BY TrackCount DESC;

    Question: List all artists who have more than 5 albums.
    SQL Query: SELECT a."Name" AS ArtistName, COUNT(al."AlbumId") AS AlbumCount FROM "Artist" a JOIN "Album" al ON a."ArtistId" = al."ArtistId" GROUP BY a."Name" HAVING COUNT(al."AlbumId") > 5;

    Question: What are the most popular playlists (with the most tracks)?
    SQL Query: SELECT p."Name" AS PlaylistName, COUNT(pt."TrackId") AS TrackCount FROM "Playlist" p JOIN "PlaylistTrack" pt ON p."PlaylistId" = pt."PlaylistId" GROUP BY p."Name" ORDER BY TrackCount DESC;

    Question: Find the average number of tracks per album.
    SQL Query: SELECT AVG(TrackCount) AS AvgTracksPerAlbum FROM (SELECT COUNT(*) AS TrackCount FROM "Track" GROUP BY "AlbumId") AS AlbumTracks;

    Question: List all customers who have never made a purchase.
    SQL Query: SELECT "CustomerId", "FirstName", "LastName", "Email" FROM "Customer" WHERE "CustomerId" NOT IN (SELECT DISTINCT "CustomerId" FROM "Invoice");

    Once again remember:
    Write only the SQL query and nothing else.
    Do not wrap the SQL query in any other text, not even backticks.
    Do not reply to the user, and only respond with SQL queries.

    Your turn:
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to get Natural Language Response Chain
def get_natural_language_chain(db: SQLDatabase):
    template = """
    You are a data analyst at a company. You have been provided with a SQL query and its result.
    Based on this information, generate a human-readable response. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    SQL Query: <SQL>{query}</SQL>
    SQL Response: {response}
    
    Provide a summary of the SQL results in plain language for the user.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to execute and combine chains
def get_combined_response(user_query: str, db: SQLDatabase, chat_history: list):
    st.write(chat_history)
    # First, run the SQL generation chain
    sql_chain = get_sql_chain(db)
    sql_query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history
    })

    print(sql_query)
    if is_safe_query(sql_query):
        # Execute the SQL query
        engine = create_engine(st.session_state.db_uri)
        with engine.connect() as connection:
            result_df = pd.read_sql(sql_query, connection)

        # Then, run the Natural Language chain
        nl_chain = get_natural_language_chain(db)
        natural_language_response = nl_chain.stream({
            "query": sql_query,
            "response": result_df,
            "chat_history": chat_history
        })

        return sql_query, result_df, natural_language_response
    else:
        return None

# Function to validate SQL query
def is_safe_query(sql_query):
    # Remove leading/trailing whitespace
    stripped_query = sql_query.strip()

    # Split the query by semicolon and take the first part
    first_part = stripped_query.split(';', 1)[0].strip()

    # Check if the first part starts with "SELECT"
    if not re.match(r'^select\s+', first_part, re.IGNORECASE):
        return False

    # Check for an even number of single quotes (') and double quotes (")
    if first_part.count("'") % 2 != 0 or first_part.count('"') % 2 != 0:
        return False

    # Additional validation on the first statement
    if '--' in first_part or '#' in first_part:
        return False

    if '/*' in first_part or '*/' in first_part:
        return False

    if re.search(r'\bunion\b', first_part, re.IGNORECASE):
        return False

    # Check for forbidden keywords in the first statement
    forbidden_keywords = ['drop', 'insert', 'update', 'delete', 'create', 'alter', 'truncate', 'exec', 'execute', 'xp_cmdshell']
    for keyword in forbidden_keywords:
        if re.search(r'\b' + keyword + r'\b', first_part, re.IGNORECASE):
            return False

    return True

def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Function for the general chat page
def general_mode():
    st.title("🦥 General Mode")

    # Initialize chat history in session state if not already done
    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # Display chat history
    for message in st.session_state.general_chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Input for user query
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.general_chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(get_response(user_query, st.session_state.general_chat_history))
        st.session_state.general_chat_history.append(AIMessage(content=response))


# Main function to run the Streamlit app
def main():
    if page == "General Mode":
        general_mode()
    elif page == "Database Mode":
        # Check the state of 'db' and set is_disabled accordingly
        is_disabled = st.session_state.db is None # User locked from doing things

        st.title("🐘 Database Mode")
        st.caption(f"Database {'Connected' if st.session_state.db else 'Not Connected'}")

        # Attempt to connect to the database only if the Connect button has been used
        if st.session_state.db is not None:
            try:
                create_engine(st.session_state.db_uri)
                display_table_schema(st.session_state.db_uri)
                display_sql_execution(st.session_state.db_uri)
            except Exception as e:
                st.error(f"Error: {e}")

        # Initialize chat history in session state if not already done
        if "database_chat_history" not in st.session_state:
            st.session_state.database_chat_history = [
                AIMessage(content="Hello! Feel free to ask me anything about your database."),
            ]

        # Viewing and rendering chat history
        if st.session_state.db is not None:
            # Display chat history
            for message in st.session_state.database_chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        if type(message.content) == str:
                            st.markdown(message.content)
                        else:
                            sql_query, sql_response, natural_language_response = message.content
                            st.code(sql_query)
                            st.dataframe(sql_response)
                            st.write(natural_language_response)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.markdown(message.content)

        # Input for user query
        user_query = st.chat_input("Type your message here...", disabled=is_disabled)
        if user_query is not None and user_query.strip() != "":

            st.session_state.database_chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                try:
                    if st.session_state.db is not None:
                        
                        result = get_combined_response(user_query, st.session_state.db, st.session_state.database_chat_history)
                        
                        if result is None:
                            invalid_generation = "Only single SELECT statements are allowed. The use of harmful keywords such as UNION, DROP, INSERT, UPDATE, DELETE, CREATE, ALTER, TRUNCATE, EXEC, EXECUTE, and XP_CMDSHELL, as well as comments (e.g., --, #, / /) or mismatched quotes, is prohibited."
                            st.write(invalid_generation)
                            st.session_state.database_chat_history.append(AIMessage(content=invalid_generation))
                        
                        else:
                            sql_query, sql_response, natural_language_response = result
                            st.code(sql_query)
                            st.dataframe(sql_response)
                            natural_language_full = st.write_stream(natural_language_response)
                            stored = [sql_query, sql_response.head(100), natural_language_full]
                            st.session_state.database_chat_history.append(AIMessage(content=stored))
                    
                    else:
                        st.error("Please connect to the database first.")
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
