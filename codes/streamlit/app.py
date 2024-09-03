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
    st.write_stream(get_sql_query_from_model_through_user_input())


def get_sql_query_from_model_through_user_input():
    """
    Get the SQL query from the model through user input.

    Returns:
        str: The SQL query generated by the model based on user input.
    """
    # Template for generating a natural language response
    # Based on the user's question and conversation history
    template = """
    ### Task
    Generate a SQL query to answer the following question:
    `{user_question}`

    ### Database Schema
    The query will run on a database with the following schema:
    {table_metadata_string}

    ### Answer
    Given the database schema, here is the SQL query that answers `{user_question}`:
    ```
    """

    user_question = input("Please enter your question: ")

    table_metadata_string = st.session_state.db.get_table_info()

    # Create a prompt template from the provided template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the language model with the specified model name and base URL
    llm = Ollama(model="sqlcoder", base_url=MODEL_BASE_URL, verbose=True)

    # Create a chain that processes the prompt and parses the output
    chain = prompt | llm | StrOutputParser()

    # Invoke the chain with the user's question and conversation history
    return chain.stream(
        {
            "user_question": user_question,
            "table_metadata_string": table_metadata_string,
        }
    )


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
