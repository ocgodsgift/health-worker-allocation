import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from style_css import style
import pandas as pd
import numpy as np


# Setup and styling
st.set_page_config(
    page_title="Health Workers Allocation Scenario Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

style()

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
        ["Home", "Health Workers Allocation", "Oupatient Scenario", "About", "Contact"],
        icons=["house", "app-indicator", "", "person lines fill", ""],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "white"},
            "icon": {"color": "green", "font-size": "20px"},
            "nav-link": {
                "font-size": "12px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "grey",
            },
            "nav-link-selected": {"background-color": "black"},
        },
    )


if choose == "Home":

    st.image("EdoDIDa.png", width=300)
    st.title("Welcome To EdoDida")
    st.write(
       """
       Welcome to Edo State Digital and Data Agency (EdoDiDa) Integrated Data and Analytics platform, the one-stop data and analytics playground.
       The EdoDiDa integrated data governance and decision intelligence platform delivers comprehensive data and analytics solutions designed to
       empower the Edo State government. Our platform supports evidence-based planning, promotes inclusive socio-economic development,
       and facilitates equitable resource allocation.
       """
    )


elif choose == "Health Workers Allocation":

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

        """
    )
    
    st.divider()

    st.write('#### Select Allocation Option:')
    option = st.selectbox(
        '',
        ('State-Wide Coverage',
         'LGA Coverage'),
        index=0
        )


    if option == "State-Wide Coverage":
        st.write('Your selected option is: ', option)

        df = pd.read_csv('health_workers.csv')

        df = df[['LGA','Doctors','Nurses','Population']]

        # Constants
        WHO_STANDARD = 44.5

        # Function to calculate coverage
        def calculate_coverage(doctors, nurses, population):
            doctors_per_10000 = (doctors / population) * 10000
            nurses_per_10000 = (nurses / population) * 10000
            return doctors_per_10000 + nurses_per_10000

        # Calculate initial coverage
        df['Coverage'] = df.apply(lambda row: calculate_coverage(row['Doctors'], row['Nurses'], row['Population']), axis=1)

        # Streamlit UI
        st.title('Health Workforce Density Analysis - Edo State')
        st.sidebar.header('State-wide Adjustments')

        # State-wide adjustment inputs
        state_additional_doctors = st.sidebar.number_input('Additional Doctors (State-wide)', min_value=0, value=0)
        state_additional_nurses = st.sidebar.number_input('Additional Nurses (State-wide)', min_value=0, value=0)

        # State-wide calculations
        total_population = df['Population'].sum()
        total_doctors = df['Doctors'].sum() + state_additional_doctors
        total_nurses = df['Nurses'].sum() + state_additional_nurses
        state_coverage = calculate_coverage(total_doctors, total_nurses, total_population)
        state_status = 'Meets WHO Standard' if state_coverage >= WHO_STANDARD else 'Below WHO Standard'

        # Display results
        st.subheader('Current Health Workforce Data by LGA')
        st.dataframe(df)

        st.subheader('State-Wide Coverage')
        state_data = {
            'Total Population': [total_population],
            'Total Doctors': [total_doctors],
            'Total Nurses': [total_nurses],
            'State-wide Coverage': [state_coverage],
            'Status': [state_status]
        }
        state_df = pd.DataFrame(state_data)
        st.dataframe(state_df)

        st.write(f"The state-wide coverage is {state_coverage:.2f}.")
        st.write(f"Status: {state_status}")

        if state_coverage > 100:
            st.error(f"The new coverage of {state_coverage:.2f}% exceeds 100%. Please check the input values.")
        elif state_coverage < WHO_STANDARD:
            st.error(f"The new coverage of {state_coverage:.2f}% is below the WHO standard of {WHO_STANDARD}%.")
        else:
            st.success(f"The new coverage of {state_coverage:.2f}% for meets the WHO standard of {WHO_STANDARD}%.")


    elif option == "LGA Coverage":
        st.write('Your selected option is: ', option)

        df = pd.read_csv('health_workers.csv')

        df = df[['LGA','Doctors','Nurses','Population']]

        # Constants
        WHO_STANDARD = 44.5

        # Generate random dataset

        # Function to calculate coverage percentage
        def calculate_coverage(doctors, nurses, population):
            doctors_per_10000 = (doctors / population) * 10000
            nurses_per_10000 = (nurses / population) * 10000
            return doctors_per_10000 + nurses_per_10000

        # Calculate initial coverage
        df['Coverage'] = df.apply(lambda row: calculate_coverage(row['Doctors'], row['Nurses'], row['Population']), axis=1)

        # Streamlit UI
        st.title('Health Workforce Density Analysis')
        st.sidebar.header('Adjust Health Workers')

        selected_lga = st.sidebar.selectbox('Select Local Government Area', df['LGA'])
        additional_doctors = st.sidebar.number_input('Additional Doctors', min_value=0, value=0)
        additional_nurses = st.sidebar.number_input('Additional Nurses', min_value=0, value=0)

        # Get selected LGA data
        lga_data = df[df['LGA'] == selected_lga].iloc[0]

        # Calculate new values
        new_doctors = lga_data['Doctors'] + additional_doctors
        new_nurses = lga_data['Nurses'] + additional_nurses
        new_coverage = calculate_coverage(new_doctors, new_nurses, lga_data['Population'])

        # Check WHO standard and limits
        coverage_status = 'Meets WHO Standard' if new_coverage >= WHO_STANDARD else 'Below WHO Standard'

        st.subheader("Current Health Workers Coverage")

        current_data_df = {
            'LGA': [selected_lga],
            'Population': [lga_data['Population']],
            'Current Doctors': [lga_data['Doctors']],
            'Current Nurses': [lga_data['Nurses']],
            'Current Coverage (%)': [lga_data['Coverage']],
        }

        current_data_df = pd.DataFrame(current_data_df)
        st.dataframe(current_data_df)

        # Display results
        st.subheader('Health Workers Coverage')
        updated_data = {
            'LGA': [selected_lga],
            'Population': [lga_data['Population']],
            'New Doctors': [new_doctors],
            'New Nurses': [new_nurses],
            'New Coverage (%)': [np.round(new_coverage, 2)],
            'Status': [coverage_status]
        }
        updated_df = pd.DataFrame(updated_data)
        st.dataframe(updated_df)

        if new_coverage > 100:
            st.error(f"The new coverage of {new_coverage:.2f}% for {selected_lga} exceeds 100%. Please check the input values.")
        elif new_coverage < WHO_STANDARD:
            st.error(f"The new coverage of {new_coverage:.2f}% for {selected_lga} is below the WHO standard of {WHO_STANDARD}%.")
        else:
            st.success(f"The new coverage of {new_coverage:.2f}% for {selected_lga} meets the WHO standard of {WHO_STANDARD}%.")

    st.write("""
                #### Overall Health Workers:
                To achieve good healthcare coverage, there should be at least 44.5 health workers (including doctors and nurses) for every 10,000 people.
            """)
    

elif choose == "Oupatient Scenario":
    # Load the data
    df = pd.read_csv('analysis.csv')

    # Define the features and target
    X = df[['PHC', 'Population 2022']]
    Y = df['Outpatient Attendance']

    population = df['Population 2022']

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Streamlit App
    st.title('Scenario Analysis: Impact of PHCs on Outpatient Attendance')

    # Sidebar Inputs
    st.sidebar.header('Adjust Scenario Parameters')
    phc_increase = st.sidebar.slider('Increase in PHCs (%)', 0, 100, 10)

    # LGA Selection
    selected_lga = st.sidebar.selectbox('Select LGA', df['LGA'])

    # Apply the percentage increases
    df_scenario = df.copy()
    df_scenario['PHC'] = df['PHC'] * (1 + phc_increase / 100)

    # Predict outpatient attendance for the scenario
    predictions_scenario = model.predict(df_scenario[['PHC', 'Population 2022']])

    for row in df['Outpatient Attendance'].index:
        if df.at[row, 'Outpatient Attendance'] - predictions_scenario[row] < 0:
            df.at[row, 'New Outpatient Attendance'] = 0
        else:
            df.at[row, 'New Outpatient Attendance'] = df.at[row, 'Outpatient Attendance'] - predictions_scenario[row]

    df_scenario['New PHC'] = np.round(df_scenario['PHC'])
    df['New Outpatient Attendance'] = np.round(df['New Outpatient Attendance'])

    # Compile the results into a dataframe
    scenario_results = pd.DataFrame({
        'LGA': df['LGA'],
        'Population': df['Population 2022'],
        'Current PHC': df['PHC'],
        'Current Outpatient Attendance': df['Outpatient Attendance'],
        'New PHC': df_scenario['New PHC'],
        'Estimated Outpatient Attendance': df['New Outpatient Attendance']
    })

    # Filter results for the selected LGA
    lga_results = scenario_results[scenario_results['LGA'] == selected_lga]

    # Calculate the reduction in outpatient attendance and percentage reduction
    lga_results['Reduction in Outpatient Attendance'] = lga_results['Current Outpatient Attendance'] - lga_results['Estimated Outpatient Attendance']
    lga_results['Reduction Percentage'] = (lga_results['Reduction in Outpatient Attendance'] / lga_results['Current Outpatient Attendance']) * 100

    # Explanation logic
    def generate_explanation(lga, current_phc, new_phc, current_attendance, new_attendance, reduction, reduction_percentage):
        new_phc = int(new_phc)
        reduction = int(reduction)
        current_attendance = int(current_attendance)
        new_attendance = int(new_attendance)
        
        explanation = f"In the LGA {lga}, increasing the number of PHCs from {current_phc} to {new_phc} results in an estimated outpatient attendance reduction from {current_attendance} to {new_attendance}. "
        
        if new_attendance == 0:
            explanation += f"This achieves a complete reduction in outpatient attendance, indicating that the increased PHCs are sufficient to meet the healthcare needs of the population."
        else:
            explanation += f"This represents a reduction of {reduction} in outpatient visits, which is a {reduction_percentage:.2f}% decrease. "
            
            if reduction_percentage > 50:
                explanation += "This substantial decrease suggests that the additional PHCs significantly improve healthcare access and reduce patient load."
            elif reduction_percentage > 20:
                explanation += "This moderate decrease indicates improved healthcare access but may suggest a need for further increases in PHCs to achieve optimal results."
            else:
                explanation += "This small decrease indicates that while the additional PHCs help, further measures might be necessary to substantially reduce outpatient attendance."
        
        return explanation

    # Generate explanation for the selected LGA
    explanation = generate_explanation(
        selected_lga,
        lga_results['Current PHC'].values[0],
        lga_results['New PHC'].values[0],
        lga_results['Current Outpatient Attendance'].values[0],
        lga_results['Estimated Outpatient Attendance'].values[0],
        lga_results['Reduction in Outpatient Attendance'].values[0],
        lga_results['Reduction Percentage'].values[0]
    )

    # Display the results
    st.subheader(f'Scenario Analysis Results for {selected_lga}')
    st.dataframe(lga_results[['LGA','Population', 'Current PHC', 'Current Outpatient Attendance', 'New PHC', 'Estimated Outpatient Attendance', 'Reduction in Outpatient Attendance']].set_index('LGA'))

    st.write(f"### Explanation for {selected_lga}")
    st.write(explanation)

    st.divider()

    # Visualization
    st.write("#### Scenario Over All Local Government Area")
    st.bar_chart(scenario_results.set_index('LGA')['Estimated Outpatient Attendance'])


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
    pass
    # # Collect users feedback
    # st.markdown(
    #     """ <style> .font {
    # font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    # </style> """,
    #     unsafe_allow_html=True,
    # )
    # st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    # # set clear_on_submit=True so that the form will be reset/cleared once
    # # it's submitted
    # with st.form(key="columns_in_form2", clear_on_submit=True):
    #     st.write(
    #         """
    #     # Please help me improve this app. Your honest feedback is highly appreciated.
    #     """
    #     )

    #     firstName = st.text_input(
    #         label="Please Enter Your First Name"
    #     )  # Collect user first name

    #     lastName = st.text_input(
    #         label="Please Enter Your Last Name"
    #     )  # Collect user last name

    #     # Collect user Email address
    #     Email = st.text_input(label="Please Enter Email")

    #     is_new_account = True

    #     resolver = caching_resolver(timeout=10)

    #     def validateEmail(email):
    #         # Helper function to validate email address
    #         try:
    #             # Check that the email address is valid.
    #             validation = validate_email(
    #                 Email, check_deliverability=is_new_account, dns_resolver=resolver
    #             )

    #             email = validation.email
    #             st.success("Thank you for your feedback.")
    #         except EmailNotValidError as e:
    #             # Email is not valid.
    #             # The exception message is human-readable.
    #             st.warning(str(e))
    #             logger.warning(
    #                 "Invalid Email. Please enter a valid email address")
    #             logger.error("Exception occurred", exc_info=True)

    #     # Collect user feedback
    #     Message = st.text_input(label="Please Enter Your Message")

    #     submitted = st.form_submit_button("Submit")

    #     if submitted:
    #         if firstName == "" or lastName == "" or Email == "" or Message == "":
    #             st.error(
    #                 "Please fill in the required details before hitting the submit button"
    #             )
    #         if Email != "":
    #             validateEmail(Email)
    #         else:
    #             st.success("Thank you for your feedback")


# the footer and more information
st.divider()
st.markdown(
    """<p style="color:white ; text-align:center;font-size:15px;">
Copyright | DSNai 2024(c)
""",
    unsafe_allow_html=True,
)

