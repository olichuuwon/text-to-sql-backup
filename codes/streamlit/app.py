"""
The script is divided into two modes: General Mode and Database Mode. 
In General Mode, the user can have a conversation with the bot and ask general questions. 
In Database Mode, the user can connect to a PostgreSQL database and perform database-related tasks.

The script uses the following libraries:
- os: for accessing environment variables
- re: for regular expression operations
- streamlit: for building the web application
- sqlalchemy: for interacting with the database
- pandas: for data manipulation and analysis
- prettytable: for creating ASCII tables
- langchain_core: for natural language processing tasks
- langchain_community: for additional utilities and models
- graphviz: for creating entity relation diagrams
"""

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
from graphviz import Digraph


# Set the Streamlit page configuration
st.set_page_config(page_title="LLM Tools", page_icon=":speech_balloon:", layout="wide")

# Load model configurations from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:instruct")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://model:11434")

DATABASE_MODE = "Database Mode"
GENERAL_MODE = "General Mode"


# Sidebar for mode selection
# Dropdown to select the mode
st.sidebar.title("LLM Tools")
page = st.sidebar.selectbox("Select Mode", [GENERAL_MODE, DATABASE_MODE])


def get_table_schema(engine, table_name: str) -> PrettyTable:
    """
    Retrieves the schema of a table from the specified database engine.
    Args:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the database.
        table_name (str): The name of the table.
    Returns:
        PrettyTable: A PrettyTable object containing the column names and data types of the table.
    """
    # Create an inspector to inspect the database
    inspector = inspect(engine)

    # Initialize a PrettyTable object to store the schema information
    schema_table = PrettyTable()
    schema_table.field_names = ["Column Name", "Data Type"]

    # Retrieve the columns of the specified table
    columns = inspector.get_columns(table_name)

    # Add each column's name and data type to the PrettyTable
    for column in columns:
        schema_table.add_row([column["name"], column["type"]])

    return schema_table


def get_sample_data(engine, table_name: str, sample_size: int = 1) -> PrettyTable:
    """
    Retrieves a sample of data from the specified table in the database.
    Args:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the database.
        table_name (str): The name of the table.
        sample_size (int): The number of rows to retrieve as a sample.
    Returns:
        PrettyTable: A PrettyTable object containing the sample data from the table.
    """
    # Construct the SQL query to retrieve a sample of data from the table
    query = f'SELECT * FROM "{table_name}" LIMIT {sample_size};'

    # Execute the query and load the result into a DataFrame
    df = pd.read_sql(query, engine)

    # Initialize a PrettyTable object to store the sample data
    sample_data_table = PrettyTable()

    # Check if the DataFrame is not empty
    if not df.empty:
        # Set the field names of the PrettyTable to the column names of the DataFrame
        sample_data_table.field_names = df.columns.tolist()

        # Add each row of the DataFrame to the PrettyTable
        for _, row in df.iterrows():
            sample_data_table.add_row(row.tolist())

    return sample_data_table


def display_table_schema(db_uri: str, sample_size: int = 1) -> None:
    """
    Display the schema and sample data of all tables in the database.

    Args:
        db_uri (str): The database URI.
        sample_size (int, optional): The number of rows to retrieve as a sample. Defaults to 1.
    """
    # Create a SQLAlchemy engine connected to the database
    engine = create_engine(db_uri)

    # Create an inspector to inspect the database
    inspector = inspect(engine)

    # Create an expander in Streamlit to view the database schema
    with st.expander("View Database Schema"):
        # Iterate over all table names in the database
        for table_name in inspector.get_table_names():
            # Display the table name
            st.write(f"**Table: {table_name}**")

            # Display the schema of the table
            st.write("Schema:")
            st.write(get_table_schema(engine, table_name))

            # Display a sample of data from the table
            st.write("Sample Data:")
            st.write(get_sample_data(engine, table_name, sample_size))


def display_entity_relation_diagram(db_uri: str):
    """
    Display the entity-relationship diagram of the database.

    Args:
        db_uri (str): The database URI.
    """
    # Create a SQLAlchemy engine connected to the database
    engine = create_engine(db_uri)

    # Fetch tables and relationships
    with st.spinner("Reading metadata..."):
        # Retrieve table information and foreign key relationships
        tables_info = list_tables_and_columns(engine)
        fk_relationships = list_foreign_keys(engine)

    # Create an expander in Streamlit to view the entity-relationship diagram
    with st.expander("View Entity Relation Diagram"):
        # Create the entity-relationship diagram
        er_diagram = create_er_diagram(tables_info, fk_relationships)
        # Display the entity-relationship diagram using Streamlit
        st.graphviz_chart(er_diagram.source, use_container_width=True)


def list_tables_and_columns(engine):
    """
    List all tables and their columns in the database.

    Args:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the database.

    Returns:
        dict: A dictionary where the keys are table names and the values are lists of column information.
    """
    with engine.connect() as connection:
        # Create an inspector to inspect the database
        inspector = inspect(connection)
        schema = "public"  # Adjust if using a different schema

        # Dictionary to store table and columns
        tables_info = {}

        # Get tables and columns
        for table_name in inspector.get_table_names(schema=schema):
            columns = []
            # Get primary keys for the table
            primary_keys = inspector.get_pk_constraint(table_name, schema=schema)[
                "constrained_columns"
            ]
            # Get column information for the table
            for column_info in inspector.get_columns(table_name, schema=schema):
                is_primary = column_info["name"] in primary_keys
                # Append column information to the list
                columns.append(
                    {
                        "name": column_info["name"],
                        "type": column_info["type"],
                        "nullable": column_info["nullable"],
                        "primary_key": is_primary,
                    }
                )
            # Add table and its columns to the dictionary
            tables_info[table_name] = columns

    return tables_info


def list_foreign_keys(engine):
    """
    List all foreign key relationships in the database.

    Args:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the database.

    Returns:
        list: A list of tuples containing foreign key relationships.
    """
    with engine.connect() as connection:
        # Create an inspector to inspect the database
        inspector = inspect(connection)
        schema = "public"  # Assuming 'public' schema; adjust if using another schema

        # List to store foreign key relationships
        fk_relationships = []

        # Get tables and their foreign keys
        for table_name in inspector.get_table_names(schema=schema):
            # Retrieve foreign keys for the table
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
            for fk in foreign_keys:
                # Append foreign key relationship to the list
                fk_relationships.append(
                    (
                        table_name,
                        fk["referred_table"],
                        fk["constrained_columns"],
                        fk["referred_columns"],
                    )
                )

    return fk_relationships


def create_er_diagram(tables_info, fk_relationships):
    """
    Create an entity-relationship diagram from table information and foreign key relationships.

    Args:
        tables_info (dict): A dictionary containing table names as keys and a list of column info as values.
        fk_relationships (list): A list of tuples containing foreign key relationships.

    Returns:
        graphviz.Digraph: A Graphviz Digraph object representing the ER diagram.
    """
    # Initialize a Graphviz Digraph object for the ER diagram
    dot = Digraph(comment="ER Diagram", format="png")

    # Create nodes for each table
    for table, columns in tables_info.items():
        # Start the HTML-like label for the table node
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        label += f'<TR><TD BGCOLOR="lightgray" COLSPAN="2"><B>{table}</B></TD></TR>'

        # Add each column to the table node
        for col in columns:
            # Format the column name and type
            col_name = (
                f"<B><U>{col['name']}</U></B>" if col["primary_key"] else col["name"]
            )
            col_name = f"*{col_name}" if col["nullable"] else col_name
            col_type = f"{col['type']}"
            label += f'<TR><TD ALIGN="LEFT">{col_name}</TD><TD ALIGN="LEFT">{col_type}</TD></TR>'

        # Close the HTML-like label for the table node
        label += "</TABLE>>"
        dot.node(table, label=label, shape="plain")

    # Create edges for foreign key relationships
    for table, ref_table, columns, ref_columns in fk_relationships:
        # Add an edge between the tables with a label for the foreign key relationship
        for col, ref_col in zip(columns, ref_columns):
            dot.edge(f"{table}", f"{ref_table}", label=f"{col} -> {ref_col}")

    return dot


def is_safe_query(sql_query):
    """
    Check if a SQL query is safe to execute by looking for potentially dangerous keywords.

    Args:
        query (str): The SQL query to check.

    Returns:
        bool: True if the query is considered safe, False otherwise.
    """
    # Remove leading/trailing whitespace
    stripped_query = sql_query.strip()

    # Define forbidden keywords and patterns
    forbidden_patterns = [
        r"\bdrop\b",
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bcreate\b",
        r"\balter\b",
        r"\btruncate\b",
        r"\bexec\b",
        r"\bexecute\b",
        r"\bxp_cmdshell\b",
        r"\bunion\b",
        r"\bjoin\b",
        r"--",
        r"#",
        r"/\*",
        r"\*/",
    ]

    # Check for forbidden patterns
    for pattern in forbidden_patterns:
        if re.search(pattern, stripped_query, re.IGNORECASE):
            return False

    # Check for an even number of single quotes (') and double quotes (")
    if stripped_query.count("'") % 2 != 0 or stripped_query.count('"') % 2 != 0:
        return False

    # Check to ensure only single statement provided
    if stripped_query.count(";") > 1:
        return False

    # Split the query by semicolon and take the first part
    first_part = stripped_query.split(";", 1)[0].strip()

    # Check if the first part starts with "SELECT"
    if not re.match(r"^select\s+", first_part, re.IGNORECASE):
        return False

    return True


def display_sql_execution(db_uri: str) -> None:
    """
    Execute a SQL query and display the results using Streamlit.

    Args:
        db_uri (str): The database URI.

    Returns:
        None
    """
    # Create a SQLAlchemy engine connected to the database
    engine = create_engine(db_uri)

    # Create an expander in Streamlit to run SQL statements manually
    with st.expander("Run Statements Manually"):
        # Display example SQL queries
        st.caption("Name 10 tracks.")
        st.code('SELECT * FROM "Track" LIMIT 10;', language="sql", line_numbers=False)

        st.caption(
            "List of the top 50 customers based on their total sales, ordered in descending order."
        )
        st.code(
            'SELECT "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName", (SELECT SUM("Invoice"."Total") FROM "Invoice" WHERE "Invoice"."CustomerId" = "Customer"."CustomerId") AS TotalSales FROM "Customer" ORDER BY TotalSales DESC LIMIT 50;',
            language="sql",
            line_numbers=False,
        )

        # Text area for user to enter SQL query
        sql_query = st.text_area(
            "Enter SQL query:", height=100, key="sql_query_sidebar"
        )

        # Button to execute the SQL query
        if st.button("Fetch", key="run_sql_sidebar"):
            # Consider only the first part of the query (up to the first semicolon)
            if is_safe_query(sql_query):
                try:
                    first_part = sql_query.split(";", 1)[0].strip()
                    with engine.connect() as connection:
                        # Execute the query and fetch the results
                        result = connection.execute(text(first_part))
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())

                        # Display the query result as a dataframe
                        st.write("Query Result:")
                        st.dataframe(df)
                except Exception as e:
                    # Display an error message if the query execution fails
                    st.error(f"Error executing SQL query: {e}")
            else:
                # Display an error message if the query is not safe
                st.error(
                    "Only single SELECT statements are allowed. The use of harmful keywords such as UNION, DROP, INSERT, UPDATE, DELETE, CREATE, ALTER, TRUNCATE, EXEC, EXECUTE, and XP_CMDSHELL, as well as comments (e.g., --, #, / /) or mismatched quotes, is prohibited."
                )


def get_sql_chain(db: SQLDatabase):
    """
    Generates an SQL query based on a user's question and the conversation history.

    Args:
        db (SQLDatabase): The SQL database object.

    Returns:
        RunnablePassthrough: The SQL query generator pipeline.
    """
    # Template for generating SQL queries based on user questions and conversation history
    template = """
    ### Instructions:
    Your task is to convert a question into a SQL query, given a Postgres database schema.
    Adhere to these rules:
    1. The current time in the database is in epoch/UTC format, so remember to convert it to UTC+8.
    2. Convert epoch timestamps to UTC+8 by adding 28,800 seconds (8 hours) to the epoch value. This will adjust the time from UTC to the desired UTC+8 timezone.
    3. Prioritize the use of `SELECT` statements with explicit column names.
    4. Avoid the use of `UNION` and `JOIN` operations. Structure queries to work without these constructs to maintain simplicity and efficiency.
    5. Limit the result set to a maximum of 1000 rows to prevent large data returns.
    6. Use Table Aliases to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    7. Avoid using constructs not supported by PostgreSQL, such as `CROSS APPLY`.
    8. Handle null values and ensure to include conditions that maintain data integrity in filters.
    9. Ensure that aggregation functions (e.g., `COUNT`, `SUM`, `AVG`) are used with `GROUP BY` clauses when necessary.
    10. The syntax for the random function in PostgreSQL is: `random()`. 
    11. For geospatial distance calculations, ONLY use the Haversine formula with standard SQL functions such as `COS`, `SIN`, `RADIANS`, and `ACOS`.
    12. When calculating distances, use 6371 as the Earth's radius in kilometers.
    13. Calculate the distance directly in the SQL query using the Haversine formula. Do not use external libraries or functions.
    
    Haversine Formula:
        6371 * ACOS(
        COS(RADIANS(1.3521)) * COS(RADIANS(latitude_column)) * COS(RADIANS(longitude_column) - RADIANS(103.8198)) +
        SIN(RADIANS(1.3521)) * SIN(RADIANS(latitude_column))

    <SCHEMA>{schema}</SCHEMA>

    ### Input:
    Format:
    Write only the SQL query and nothing else.
    Do not wrap the SQL query in any other text, not even backticks.
    Do not reply to the user, and only respond with SQL queries.

    For example:

    Question: Find all records from the bird_locations table where the bird's location is within an 8888 km radius of a specific home location (Latitude: 1.3521, Longitude: 103.8198).
    SQL Query: SELECT * FROM bird_locations WHERE ( 6371 * ACOS( COS(RADIANS(1.3521)) * COS(RADIANS(latitude)) * COS(RADIANS(longitude) - RADIANS(103.8198)) + SIN(RADIANS(1.3521)) * SIN(RADIANS(latitude)) ) ) <= 8888;

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
    SQL Query: SELECT *, "InvoiceDate" AT TIME ZONE 'UTC' AS "InvoiceDate_UTC" FROM "Invoice" ORDER BY "InvoiceDate_UTC" DESC LIMIT 5;

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
    # Create a prompt template from the provided template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the language model with the specified model name and base URL
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    def get_schema(_):
        """
        Retrieve the schema information from the database.

        Args:
            _ : Placeholder argument for compatibility.

        Returns:
            dict: The schema information of the database.
        """
        return db.get_table_info()

    # Create a runnable chain that assigns the schema, processes the prompt, and parses the output
    return (
        RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
    )


def get_natural_language_chain(db: SQLDatabase):
    """
    Create a chain to generate a natural language response based on a SQL query and its result.

    Args:
        db (SQLDatabase): The SQLDatabase object to interact with the database.

    Returns:
        Runnable: A runnable chain that processes the SQL query
        and its result to generate a human-readable response.
    """
    # Template for generating a human-readable response
    template = """
    You are a data analyst at a company. You have been provided with a SQL query and its result.
    Based on this information, generate a human-readable response. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    SQL Query: <SQL>{query}</SQL>
    SQL Response: {response}
    
    Provide a summary of the SQL results in plain language for the user.
    """

    # Create a prompt template from the provided template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the language model with the specified model name and base URL
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    def get_schema(_):
        """
        Retrieve the schema information from the database.

        Args:
            _ : Placeholder argument for compatibility.

        Returns:
            dict: The schema information of the database.
        """
        return db.get_table_info()

    # Create a runnable chain that assigns the schema, processes the prompt, and parses the output
    return (
        RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
    )


def get_combined_response(user_query: str, db: SQLDatabase, chat_history: list):
    """
    Generate an SQL query based on the user's question,
    execute it, and provide a natural language response.

    Args:
        user_query (str): The user's question.
        db (SQLDatabase): The SQL database object.
        chat_history (list): The conversation history.

    Returns:
        tuple: A tuple containing the SQL query,
        the result DataFrame,
        and the natural language response.

    Returns None if the query is not safe.
    """
    with st.spinner():
        # First, run the SQL generation chain
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke(
            {"question": user_query, "chat_history": chat_history}
        )

        print(sql_query)
        if is_safe_query(sql_query):
            # Execute the SQL query
            engine = create_engine(st.session_state.db_uri)
            with engine.connect() as connection:
                result_df = pd.read_sql(sql_query, connection)

            # Then, run the Natural Language chain
            nl_chain = get_natural_language_chain(db)
            natural_language_response = nl_chain.stream(
                {
                    "query": sql_query,
                    "response": result_df,
                    "chat_history": chat_history,
                }
            )

            return sql_query, result_df, natural_language_response
        return None


def get_response(user_query, chat_history):
    """
    Generate a natural language response based on the user's question and the conversation history.

    Args:
        user_query (str): The user's question.
        chat_history (list): The conversation history.

    Returns:
        str: The natural language response.
    """
    # Template for generating a natural language response
    # Based on the user's question and conversation history
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    # Create a prompt template from the provided template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the language model with the specified model name and base URL
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    # Create a chain that processes the prompt and parses the output
    chain = prompt | llm | StrOutputParser()

    # Invoke the chain with the user's question and conversation history
    return chain.stream(
        {
            "chat_history": chat_history,
            "user_question": user_query,
        }
    )


def general_mode_function():
    """
    Function for the general chat page in the Streamlit app.

    This function initializes the chat history,
    displays the chat messages,
    and handles user input to generate responses.

    Returns:
        None
    """
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
        # Append the user's message to the chat history
        st.session_state.general_chat_history.append(HumanMessage(content=user_query))

        # Display the user's message
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate and display the AI's response
        with st.chat_message("AI"):
            response = st.write_stream(
                get_response(user_query, st.session_state.general_chat_history)
            )
        # Append the AI's response to the chat history
        st.session_state.general_chat_history.append(AIMessage(content=response))


def database_mode_function():
    """
    Function for the database mode page in the Streamlit app.

    This function handles the database connection settings, displays the database schema and entity-relationship diagram,
    and manages chat interactions for querying the database.

    Returns:
        None
    """
    st.title("🐘 Database Mode")
    initialize_session_state()
    display_connection_settings()
    handle_database_connection()
    display_database_info()
    initialize_chat_history()
    display_chat_history()
    handle_user_query()


def initialize_session_state():
    """
    Initializes the session state by checking if the 'db_uri' and 'db' variables are present in the session state.
    If they are not present, they are set to None.

    Parameters:
        None

    Returns:
        None
    """
    if "db_uri" not in st.session_state:
        st.session_state.db_uri = None
    if "db" not in st.session_state:
        st.session_state.db = None


def display_connection_settings():
    """
    Displays the current database connection settings and allows the user to modify them.
    Returns:
        None
    """
    st.caption(f"Database {'Connected' if st.session_state.db else 'Not Connected'}")
    st.sidebar.title("Database Connection Settings")
    db_host = st.sidebar.text_input("Host", value=os.getenv("DB_HOST", "postgres"))
    db_port = st.sidebar.text_input("Port", value=os.getenv("DB_PORT", "5432"))
    db_user = st.sidebar.text_input("User", value=os.getenv("DB_USER", "user"))
    db_password = st.sidebar.text_input(
        "Password", value=os.getenv("DB_PASSWORD", "pass"), type="password"
    )
    db_name = st.sidebar.text_input("Database", value=os.getenv("DB_NAME", "chinook"))

    st.session_state.db_uri = uri_database(
        db_user, db_password, db_host, db_port, db_name
    )


def uri_database(user: str, password: str, host: str, port: str, database: str) -> str:
    """
    Generates a PostgreSQL URI based on the provided parameters.

    Args:
        user (str): The username for the database connection.
        password (str): The password for the database connection.
        host (str): The host address of the database server.
        port (str): The port number of the database server.
        database (str): The name of the database.

    Returns:
        str: The PostgreSQL URI for the database connection.
    """
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def handle_database_connection():
    """
    Handles the database connection based on the user's input.

    Returns:
        None
    """
    if st.sidebar.button("Connect"):
        try:
            engine = create_engine(st.session_state.db_uri)
            with engine.connect():
                st.sidebar.success("Successfully connected to the database!")
                st.session_state.db = SQLDatabase.from_uri(st.session_state.db_uri)
        except Exception as e:
            st.sidebar.error(f"Failed to connect to database: {e}")


def display_database_info():
    """
    Displays information about the database.

    If the `db` attribute in the `st.session_state` is not None, this function
    will attempt to display the SQL execution, table schema, and entity relation
    diagram for the database specified by the `db_uri` attribute in the
    `st.session_state`. If any exception occurs during the process, an error
    message will be displayed.

    Raises:
        Exception: If any error occurs during the process of displaying the
            database information.

    """
    if st.session_state.db is not None:
        try:
            display_sql_execution(st.session_state.db_uri)
            display_table_schema(st.session_state.db_uri)
            display_entity_relation_diagram(st.session_state.db_uri)
        except Exception as e:
            st.error(f"Error: {e}")


def initialize_chat_history():
    """
    Initializes the chat history for the database.

    If the 'database_chat_history' key is not present in the session state, it adds an initial message to the chat history.

    Parameters:
        None

    Returns:
        None
    """
    if "database_chat_history" not in st.session_state:
        st.session_state.database_chat_history = [
            AIMessage(
                content="Hello! Feel free to ask me anything about your database."
            ),
        ]


def display_chat_history():
    """
    Displays the chat history in the Streamlit app.

    This function iterates through the chat history stored in the `database_chat_history` list
    and displays each message in the Streamlit app. The messages are categorized as either
    AI messages or Human messages.

    AI messages are displayed with the label "AI" and their content is rendered using Markdown.
    If the content is a string, it is displayed as is. If the content is a tuple containing a SQL query
    and a natural language response, the SQL query is displayed as code and the natural language
    response is displayed as plain text.

    Human messages are displayed with the label "Human" and their content is rendered using Markdown.

    Note:
    - The chat history is stored in the `database_chat_history` list.
    - The `st` object is assumed to be available and is used to display the messages in the Streamlit app.
    - The `AIMessage` and `HumanMessage` classes are assumed to be defined elsewhere.

    Example usage:
    ```
    display_chat_history()
    ```
    """
    if st.session_state.db is not None:
        for message in st.session_state.database_chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    if isinstance(message.content, str):
                        st.markdown(message.content)
                    else:
                        sql_query, natural_language_response = message.content
                        st.code(sql_query)
                        st.write(natural_language_response)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)


def handle_user_query():
    """
    Handles the user query in the chat interface.
    This function takes the user's input message and processes it to generate a response. It performs the following steps:
    1. Checks if the database connection is available.
    2. Appends the user's message to the chat history.
    3. Displays the user's message in the chat interface.
    4. Generates a response using the `get_combined_response` function.
    5. Checks if the response is valid.
    6. Displays the SQL query, SQL response, and natural language response in the chat interface.
    7. Appends the generated response to the chat history.
    If the database connection is not available, an error message is displayed.
    Raises:
        Exception: If there is an error generating the response.
    """
    # code implementation goes here
    is_disabled = st.session_state.db is None
    user_query = st.chat_input("Type your message here...", disabled=is_disabled)
    if user_query is not None and user_query.strip() != "":
        st.session_state.database_chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            try:
                if st.session_state.db is not None:
                    result = get_combined_response(
                        user_query,
                        st.session_state.db,
                        st.session_state.database_chat_history,
                    )

                    if result is None:
                        invalid_generation = "Only single SELECT statements are allowed. The use of harmful keywords such as UNION, DROP, INSERT, UPDATE, DELETE, CREATE, ALTER, TRUNCATE, EXEC, EXECUTE, and XP_CMDSHELL, as well as comments (e.g., --, #, / /) or mismatched quotes, is prohibited."
                        st.write(invalid_generation)
                        st.session_state.database_chat_history.append(
                            AIMessage(content=invalid_generation)
                        )
                    else:
                        sql_query, sql_response, natural_language_response = result
                        st.code(sql_query)
                        st.dataframe(sql_response)
                        natural_language_full = st.write_stream(
                            natural_language_response
                        )
                        stored = [sql_query, natural_language_full]
                        st.session_state.database_chat_history.append(
                            AIMessage(content=stored)
                        )
                else:
                    st.error("Please connect to the database first.")
            except Exception as e:
                st.error(f"Error generating response: {e}")


# Main function to run the Streamlit app
def main():
    """
    Main function to run the Streamlit app.

    This function handles the initialization and rendering of the general mode and database mode pages.

    Returns:
        None
    """
    # Check which mode to run based on the page variable
    if page == GENERAL_MODE:
        general_mode_function()
    elif page == DATABASE_MODE:
        database_mode_function()


# Entry point of the script
if __name__ == "__main__":
    main()
