import os
import streamlit as st
from sqlalchemy import create_engine, inspect
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
st.sidebar.title("Different Chats")
page = st.sidebar.radio("Go to", ["Database Mode", "General Mode"])

# Sidebar inputs for database connection, only shown in Database Mode
if page == "Database Mode":
    st.sidebar.title("Database Connection Settings")
    db_host = st.sidebar.text_input("Host", value=os.getenv('DB_HOST', 'localhost'))
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
def display_table_info(db_uri: str, sample_size: int = 1) -> None:
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        st.write(f"**Table: {table_name}**")
        st.write("Schema:")
        st.write(get_table_schema(engine, table_name))
        st.write("Sample Data:")
        st.write(get_sample_data(engine, table_name, sample_size))

# Function to get SQL query chain
def get_sql_chain(db: SQLDatabase):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    ### Instructions:
    1. **Use UTC Timing**: Always use UTC timing for any date or time calculations.
    2. **Epoch Conversion**: Convert epoch timestamps to UTC timing if necessary.
    3. **No UNION or JOINs**: Avoid using UNION and JOIN operations in your queries.
    4. **Row Limit**: Ensure that the result set is limited to a maximum of 100,000 rows.
    
    ### Examples:
    
    - **Example 1**
        - Question: Which 3 genres have the most tracks?
        - SQL Query: `SELECT GenreId, COUNT(*) as track_count FROM "Track" GROUP BY GenreId ORDER BY track_count DESC LIMIT 3;`
    
    - **Example 2**
        - Question: Name 10 playlists.
        - SQL Query: `SELECT "Name" FROM "Playlist" LIMIT 10;`
    
    - **Example 3**
        - Question: What are the 5 most recent invoices in UTC?
        - SQL Query: `SELECT * FROM "Invoice" ORDER BY "InvoiceDate" AT TIME ZONE 'UTC' DESC LIMIT 5;`
    
    - **Example 4**
        - Question: List all customers who signed up after 1609459200 epoch time in UTC.
        - SQL Query: `SELECT "CustomerId", "FirstName", "LastName", "Email" FROM "Customer" WHERE "SignUpDate" >= to_timestamp(1609459200) AT TIME ZONE 'UTC';`
    
    - **Example 5**
        - Question: How many tracks are longer than 5 minutes?
        - SQL Query: `SELECT COUNT(*) FROM "Track" WHERE "Milliseconds" > 300000;`
    
    - **Example 6**
        - Question: Get the first 100,000 rows from the track details.
        - SQL Query: `SELECT * FROM "Track" LIMIT 100000;`
    
    - **Example 7**
        - Question: Show the total sales for each customer without using joins.
        - SQL Query: `SELECT "CustomerId", "FirstName", "LastName", (SELECT SUM("Total") FROM "Invoice" WHERE "Invoice"."CustomerId" = "Customer"."CustomerId") AS TotalSales FROM "Customer" LIMIT 100000;`
    
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

# Function to generate a response
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: print(vars["query"]) or db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

    print(f"Generated SQL Query: {result}")

    return result

# Function for the general chat page
def llama_page():
    st.title("General Mode")

    # Initialize chat history in session state if not already done
    if "llama_chat_history" not in st.session_state:
        st.session_state.llama_chat_history = [
            AIMessage(content="Hello! Feel free to ask me about anything."),
        ]

    # Display chat history
    for message in st.session_state.llama_chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Input for user query
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.llama_chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            llm = Ollama(model=MODEL_NAME, base_url=MODEL_BASE_URL, verbose=True)
            response = llm(user_query)
            st.markdown(response)

        st.session_state.llama_chat_history.append(AIMessage(content=response))

# Main function to run the Streamlit app
def main():
    if page == "Database Mode":
        st.title("Database Mode")

        # Attempt to connect to the database only if the Connect button has been used
        if st.session_state.db is not None:
            try:
                engine = create_engine(st.session_state.db_uri)
                display_table_info(st.session_state.db_uri)
            except Exception as e:
                st.error(f"Error: {e}")

        # Initialize chat history in session state if not already done
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! Feel free to ask me anything about your database."),
            ]

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        # Input for user query
        user_query = st.chat_input("Type a message...")
        if user_query is not None and user_query.strip() != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                try:
                    if st.session_state.db is not None:
                        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                        st.markdown(response)
                        st.session_state.chat_history.append(AIMessage(content=response))
                    else:
                        st.error("Please connect to the database first.")
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    elif page == "General Mode":
        llama_page()

if __name__ == "__main__":
    main()
