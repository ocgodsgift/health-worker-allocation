import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
from style_css import style
import pandas as pd
import numpy as np
from email_validator import validate_email, EmailNotValidError, caching_resolver


# Setup and styling
st.set_page_config(
    page_title="Health Workers Allocation Scenario Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


with st.sidebar:
    choose = option_menu(
        "Main Menu",
        ["Homepage", "Scenario Analysis", "About", "Contact"],
        icons=["house", "app-indicator", "information_source", "person lines fill"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#008000"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#ADD8E6",
            },
            "nav-link-selected": {"background-color": "#00008B"},
        },
    )


if choose == "Homepage":
    st.markdown(
        """ <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">Health Worker Coverage Scenario Analysis</p>',
                unsafe_allow_html=True)

    st.write(
        """
        #### What is Health Workforce Density?
        Health Workforce Density means the number of healthcare workers (like doctors, nurses, and midwives) available to take care of people in a given area. It's measured by counting how many of these workers are available for every 10,000 people in the population.

        #### Why is it Important?
        Imagine you're in a town, and there are only a few doctors, nurses, and midwives available. If many people get sick at the same time, there won't be enough healthcare workers to take care of everyone quickly and effectively. Having an adequate number of healthcare workers ensures that people can get medical help when they need it, without long waits or shortages.

        #### How is it Measured?
        We look at how many doctors, nurses, and midwives there are for every 10,000 people. This gives us a good idea of whether there are enough healthcare workers to meet the needs of the community.

        #### WHO Recommendations
        The World Health Organization (WHO) gives us some guidelines on the minimum number of healthcare workers needed to ensure good healthcare services:

        Doctors: At least 1 doctor for every 1,000 people. This means for every 10,000 people, there should be at least 10 doctors.
        Nurses: At least 3 nurses for every 1,000 people. So, for every 10,000 people, there should be at least 30 nurses.

        #### Overall Health Workers:
        To achieve good healthcare coverage, there should be at least 44.5 health workers (including doctors and nurses) for every 10,000 people.
        """
    )


elif choose == "Scenario Analysis":

    option = st.selectbox(
        'Which scraping option will you prefer?',
        ('State-Wide Coverage',
         'Coverage by LGA'),
        index=0
        )


    if option == "State-Wide Coverage":
        pass


    if option == "Coverage by LGA":
        st.write('Your selected option is: ', option)
        # Load your dataset
        model_df = pd.read_csv('health_workers.csv')

        # Initialize and train the model
        from sklearn.model_selection import train_test_split
        scaler = StandardScaler()
        model = LinearRegression()
        X = model_df[['Doctors', 'Nurses', 'Population']].values
        y = model_df['Total Coverage (WHC) %'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)

        # Function to predict new coverage
        def predict_new_coverage(lga, additional_doctors, additional_nurses):
            LGA = model_df[model_df['LGA'] == lga]
            current_doctors = LGA['Doctors'].values[0]
            current_nurses = LGA['Nurses'].values[0]
            population = LGA['Population'].values[0]

            new_doctors = current_doctors + additional_doctors
            new_nurses = current_nurses + additional_nurses

            X_new = np.array([[new_doctors, new_nurses, population]])
            new_coverage = model.predict(X_new)[0]

            return new_coverage

        st.divider()

        lga = st.selectbox('Select LGA', model_df['LGA'].unique())
        additional_doctors = st.number_input(
            'Additional Doctors', min_value=0, max_value=100, step=1)
        additional_nurses = st.number_input(
            'Additional Nurses', min_value=0, max_value=100, step=1)

        if st.button('Predict Total Coverage'):
            new_coverage = predict_new_coverage(
                lga, additional_doctors, additional_nurses)
            st.write(f'Predicted Total Coverage for {lga}: {new_coverage: .2f}%')


elif choose == "About":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:  # To display the header text using css style
        st.markdown(
            """ <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="font">About DSN</p>',
                    unsafe_allow_html=True)
    # with col2:               # To display brand log
    #    st.image(logo, width=130, caption="Twitter Logo")

    # st.write(""""")
    # st.image(profile, width=200, caption="Muhammad's Profile Picture")


elif choose == "Contact":
    # Collect users feedback
    st.markdown(
        """ <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    # set clear_on_submit=True so that the form will be reset/cleared once
    # it's submitted
    with st.form(key="columns_in_form2", clear_on_submit=True):
        st.write(
            """
        # Please help me improve this app. Your honest feedback is highly appreciated.
        """
        )

        firstName = st.text_input(
            label="Please Enter Your First Name"
        )  # Collect user first name

        lastName = st.text_input(
            label="Please Enter Your Last Name"
        )  # Collect user last name

        # Collect user Email address
        Email = st.text_input(label="Please Enter Email")

        is_new_account = True

        resolver = caching_resolver(timeout=10)

        def validateEmail(email):
            # Helper function to validate email address
            try:
                # Check that the email address is valid.
                validation = validate_email(
                    Email, check_deliverability=is_new_account, dns_resolver=resolver
                )

                email = validation.email
                st.success("Thank you for your feedback.")
            except EmailNotValidError as e:
                # Email is not valid.
                # The exception message is human-readable.
                st.warning(str(e))
                logger.warning(
                    "Invalid Email. Please enter a valid email address")
                logger.error("Exception occurred", exc_info=True)

        # Collect user feedback
        Message = st.text_input(label="Please Enter Your Message")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if firstName == "" or lastName == "" or Email == "" or Message == "":
                st.error(
                    "Please fill in the required details before hitting the submit button"
                )
            if Email != "":
                validateEmail(Email)
            else:
                st.success("Thank you for your feedback")


# the footer and more information
st.info(
    "HELP : You can reach out to me via EMAIL below if you need a simple WEB AUTOMATION for your organization. Thank you for using my Twitter scraping app"
)
st.write("")
st.markdown(
    """<p style="color:white ; text-align:center;font-size:15px;">
Copyright | DSNai 2024(c)
""",
    unsafe_allow_html=True,
)
st.markdown(
    """<p style="color:white; text-align:center;font-size:15px;">
📞+2348108316393
""",
    unsafe_allow_html=True,
)
