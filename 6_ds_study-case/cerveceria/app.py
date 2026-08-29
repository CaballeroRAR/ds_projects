import streamlit as st
import pandas as pd
import datetime
from google.cloud import bigquery

PROJECT_ID = "g-prometheus"
DATASET_ID = "cerveceria_data"
TABLE_ID = "quiero_chela"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

st.set_page_config(
    page_title="Quiero Chela User Administrator",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 60% !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def fetch_all_users() -> pd.DataFrame:
    client = get_bq_client()
    query = f"SELECT ID, NAME, EMAIL, DOB, PHONE FROM `{TABLE_REF}` ORDER BY ID ASC"
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error al consultar BigQuery: {e}")
        return pd.DataFrame(columns=["ID", "NAME", "EMAIL", "DOB", "PHONE"])

def insert_user_to_bq(user_data: dict) -> bool:
    client = get_bq_client()
    schema = [
        bigquery.SchemaField("ID", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("NAME", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("EMAIL", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("DOB", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("PHONE", "STRING", mode="NULLABLE"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    try:
        job = client.load_table_from_json([user_data], TABLE_REF, job_config=job_config)
        job.result()
        return True
    except Exception as e:
        st.error(f"Error al insertar usuario en BigQuery: {e}")
        return False

st.title("Quiero Chela User Administrator")

st.sidebar.title("Navegación")
menu_option = st.sidebar.radio(
    "Seleccione una opción:",
    options=["Añadir Nuevo Usuario", "Ver / Buscar Usuarios"],
    index=0
)

df_users = fetch_all_users()

if menu_option == "Añadir Nuevo Usuario":
    st.header("Añadir Nuevo Usuario")

    if not df_users.empty and "ID" in df_users.columns:
        next_id = int(df_users["ID"].max()) + 1
    else:
        next_id = 0

    st.info(f"ID (Automático): **{next_id}**")

    with st.form(key="add_user_form", clear_on_submit=True):
        name = st.text_input("Nombre")
        email = st.text_input("Email")
        dob_date = st.date_input("Fecha de Nacimiento", value=datetime.date(1996, 1, 1))
        phone = st.text_input("Teléfono (10 números)")

        submit_button = st.form_submit_button(label="Añadir Usuario")

        if submit_button:
            if not name.strip():
                st.error("El nombre es requerido.")
            else:
                dob_str = dob_date.strftime("%d/%m/%Y")
                new_user = {
                    "ID": next_id,
                    "NAME": name.strip(),
                    "EMAIL": email.strip(),
                    "DOB": dob_str,
                    "PHONE": phone.strip()
                }

                if insert_user_to_bq(new_user):
                    st.success(f"Usuario '{name}' añadido exitosamente con ID {next_id}.")
                    st.rerun()

elif menu_option == "Ver / Buscar Usuarios":
    st.header("Usuarios Registrados")

    search_query = st.text_input("Buscar (ID, Nombre o Teléfono):", value="").strip()

    if not search_query:
        st.dataframe(df_users, use_container_width=True, hide_index=True)
    else:
        q_lower = search_query.lower()
        filtered_df = df_users[
            df_users["ID"].astype(str).str.lower().str.contains(q_lower, na=False) |
            df_users["NAME"].astype(str).str.lower().str.contains(q_lower, na=False) |
            df_users["PHONE"].astype(str).str.lower().str.contains(q_lower, na=False)
        ]

        if filtered_df.empty:
            st.warning("No se encontraron resultados.")
        else:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
