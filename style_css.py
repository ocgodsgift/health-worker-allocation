import streamlit as st

# Apply custom CSS

def style():
    
    st.markdown(
            """
        <style>
            .main {
                max-width: 90%;
                margin-left: auto;
                margin-right: auto;
                margin-top: 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                
            }

        </style>
            """,
            unsafe_allow_html=True
        )