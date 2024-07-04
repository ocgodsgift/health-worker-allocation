import streamlit as st
from streamlit_option_menu import option_menu
from style_css import style
import pandas as pd
import numpy as np
from slideshow import slideshow

# Setup and styling
st.set_page_config(
    page_title="Scenario Analysis",
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
        ["Home", "Health", "Education", "About", "Contact"],
        icons=["house", "heart", "book", "pen", "phone"],
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
    st.divider()
    slideshow()


elif choose == "Health":

    option = st.selectbox(
        '',
        ('Health Worker Allocation',
         'Health Insurance',
         'OutPatient Scenario'
         ),
        index=0
        )

    if option == "Health Worker Allocation":

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

        # Load dataset
        df = pd.read_csv('health_workers.csv')
        df = df[['LGA', 'Doctors', 'Nurses', 'Population']]

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
        # Sidebar state-wide adjustment inputs
        st.sidebar.header('State-wide Adjustments')
        state_additional_doctors = st.sidebar.number_input('Additional Doctors (State-wide)', min_value=0, value=0)
        state_additional_nurses = st.sidebar.number_input('Additional Nurses (State-wide)', min_value=0, value=0)

        # Add a button to trigger the calculation
        if st.sidebar.button('Calculate New Coverage'):
            # Distribute additional workforce proportionally based on population
            df['Additional Doctors'] = (df['Population'] / df['Population'].sum() * state_additional_doctors).astype(int)
            df['Additional Nurses'] = (df['Population'] / df['Population'].sum() * state_additional_nurses).astype(int)
            
            # Calculate new coverage with proposed additions
            df['New Doctors'] = df['Doctors'] + df['Additional Doctors']
            df['New Nurses'] = df['Nurses'] + df['Additional Nurses']
            df['New Coverage'] = df.apply(lambda row: calculate_coverage(row['New Doctors'], row['New Nurses'], row['Population']), axis=1)

            # State-wide calculations
            total_population = df['Population'].sum()
            total_doctors = df['New Doctors'].sum()
            total_nurses = df['New Nurses'].sum()
            state_coverage = calculate_coverage(total_doctors, total_nurses, total_population)
            state_status = 'Meets WHO Standard' if state_coverage >= WHO_STANDARD else 'Below WHO Standard'

            # Display results
            st.write('#### State-Wide Coverage with Proposed Additions')
            state_data = {
                'Total Population': [total_population],
                'Total Doctors': [total_doctors],
                'Total Nurses': [total_nurses],
                'State-wide Coverage': [np.round(state_coverage, 2)],
                'Status': [state_status]
            }
            state_df = pd.DataFrame(state_data)
            st.dataframe(state_df.set_index('Total Population'))

            # Display state-wide coverage status
            st.write('#### State-Wide Coverage with Proposed Additions')
            if state_coverage > 100:
                st.error(f"The new coverage of {state_coverage:.2f}% exceeds 100%. Please check the input values.")
            elif state_coverage < WHO_STANDARD:
                st.error(f"The new coverage of {state_coverage:.2f}% is below the WHO standard of {WHO_STANDARD}%.")
            else:
                st.success(f"The new coverage of {state_coverage:.2f}% meets the WHO standard of {WHO_STANDARD}%.")

            # Display health workforce data by LGA with proposed additions
            df['Status'] = df['New Coverage'].apply(lambda x: 'Meets WHO Standard' if x >= WHO_STANDARD else 'Below WHO Standard')

            def color_status(val):
                color = 'green' if val == 'Meets WHO Standard' else 'red'
                return f'color: {color}'

            st.write('#### Health Workforce Data by LGA with Additions')
            st.dataframe(df[['LGA', 'Population', 'Doctors', 'Nurses', 'Additional Doctors', 'Additional Nurses', 'New Doctors', 'New Nurses', 'New Coverage', 'Status']].set_index('LGA').style.applymap(color_status, subset=['Status']))
       
        else:
            total_population = df['Population'].sum()
            total_doctors = df['Doctors'].sum()
            total_nurses = df['Nurses'].sum()
            state_coverage = calculate_coverage(total_doctors, total_nurses, total_population)
            state_status = 'Meets WHO Standard' if state_coverage >= WHO_STANDARD else 'Below WHO Standard'

            state_data = {
                'Total Population': [total_population],
                'Total Doctors': [total_doctors],
                'Total Nurses': [total_nurses],
                'State-wide Coverage': [np.round(state_coverage, 2)],
                'Status': [state_status]
            }

            state_df = pd.DataFrame(state_data)

            # Displapy initial health workfoce state-wide
            st.write('#### State Wide Workforce')
            st.dataframe(state_df.set_index('Total Population'))

            # Display initial health workforce data by LGA
            st.write('#### Health Workforce Data by LGA')
            st.dataframe(df[['LGA', 'Population', 'Doctors', 'Nurses', 'Coverage']].set_index('LGA'))

        st.write("""
                #### Overall Health Workers:
                To achieve good healthcare coverage, there should be at least 44.5 health workers (including doctors and nurses) for every 10,000 people.
                """)

    elif option == "OutPatient Scenario":

        from sklearn.ensemble import RandomForestRegressor
        # Load the data
        df = pd.read_csv('outpateince.csv')

        # Define the features and target
        X = df[['PHC', 'Population 2022']]
        Y = df['Outpatient Attendance']

        population = df['Population 2022']

        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, Y)

        # Streamlit App
        st.write('### Scenario Analysis: Impact of PHCs on Outpatient Attendance')

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

    elif option == "Health Insurance":

        df = pd.read_excel('health_insurance.xlsx')
        from sklearn.linear_model import LinearRegression

        # Sidebar title and header
        st.sidebar.title("EHIC Scenario Analysis")
        #st.sidebar.header("Select LGA and Adjust Enrollment Rate")

        # Dropdown to select LGA
        selected_lga = st.sidebar.selectbox("Select LGA", df['LGA'])

        # Slider for enrollment rate adjustment
        adjustment = st.sidebar.slider("Enrollment Rate Adjustment (%)", -50, 50, 5, 5)

        # Predict function
        def predict_metrics(enrollment_rate_increase):
            X = df[['enrollment_rate']]
            y_death = df['death_rate']
            y_inpatient = df['In-patient']
            y_outpatient = df['OutPatient']

            # Initialize Linear Regression models
            model_death = LinearRegression()
            model_inpatient = LinearRegression()
            model_outpatient = LinearRegression()

            # Train models
            model_death.fit(X, y_death)
            model_inpatient.fit(X, y_inpatient)
            model_outpatient.fit(X, y_outpatient)

            # Predictions for the specified enrollment rate increase
            predicted_death_rate = model_death.predict([[enrollment_rate_increase]])
            predicted_inpatient = model_inpatient.predict([[enrollment_rate_increase]])
            predicted_outpatient = model_outpatient.predict([[enrollment_rate_increase]])

            return predicted_death_rate[0], predicted_inpatient[0], predicted_outpatient[0]

        # Main content area
        st.subheader("EHIC Scenario Analysis")

        # Explanation of the scenario analysis
        st.write("""
                #### Objective:
                The EHIC Scenario Analysis is designed to evaluate the implications or effects of increasing the health insurance enrollment rate on various health metrics, such as death rate, inpatients, and outpatients per Local Government Area (LGA). By adjusting the enrollment rate, we can predict and analyze how changes in health insurance coverage can impact the overall health outcomes in different regions. This allows Edo State Government and stakeholders to make informed decisions based on data-driven insights.
                 """)

        # Filter data for the selected LGA
        lga_data = df[df['LGA'] == selected_lga]

        # Display original and adjusted data
        st.write(f"#### Selected LGA: {selected_lga}")
        st.dataframe(lga_data.set_index("LGA"))

        # Calculate adjusted values based on slider input
        adjusted_enrollment_rate = lga_data['enrollment_rate'] * (1 + adjustment / 100)
        lga_data['adjusted_enrollment_rate'] = adjusted_enrollment_rate

        # Prediction and display based on slider input
        predicted_death, predicted_inpatient, predicted_outpatient = predict_metrics(1 + adjustment / 100)

        # Display predicted metrics as a dataframe
        st.write("#### Predicted Metrics")
        predicted_data = {
            "Metric": ["In-patient", "Out-patient", "Death Rate"],
            "Outcomes": [round(predicted_inpatient, 2), round(predicted_outpatient, 2), round(predicted_death, 2)]
        }
        predicted_df = pd.DataFrame(predicted_data)
        predicted_df = predicted_df.set_index("Metric")
        st.dataframe(predicted_df.T)

elif choose == "Education":
    # Load the data
    ta_df = pd.read_excel("teachers_allocation.xlsx")

    # Group by 'LGA' and calculate the sum of 'total_students' and 'total_teachers' for each 'LGA'
    lga_ta = ta_df.groupby('LGA').agg({'total_students': 'sum', 'total_teachers': 'sum'}).reset_index()

    # Calculate the ideal number of students per teacher
    ideal_students_per_teacher = 15

    # Calculate the actual number of students per teacher in each LGA
    lga_ta['actual_students_per_teacher'] = round(lga_ta['total_students'] / lga_ta['total_teachers'], 0)

    # Calculate the percentage coverage in each LGA
    lga_ta['percentage_coverage'] = round((ideal_students_per_teacher / lga_ta['actual_students_per_teacher']) * 100, 2)

    # Function to calculate percentage coverage
    def calculate_coverage(df, lga, additional_teachers, additional_students):
        ideal_students_per_teacher = 15
        lga_row = df[df['LGA'] == lga]
        new_total_teachers = lga_row['total_teachers'].values[0] + additional_teachers
        new_total_students = lga_row['total_students'].values[0] + additional_students
        new_actual_students_per_teacher = new_total_students / new_total_teachers
        new_percentage_coverage = round((ideal_students_per_teacher / new_actual_students_per_teacher) * 100, 2)
        return new_percentage_coverage

    # Streamlit interface
    st.subheader("Teachers Allocation Scenario Analysis")
    st.write("""
             Welcome to the Teachers Allocation Scenario Analysis application!") 
             This tool is designed to help education administrators and policymakers analyze and optimize the allocation of
             senior secondary schoo (SSS) teachers across the 18 Local Government Areas (LGAs) in Edo State.""")

    # User inputs
    lga = st.sidebar.selectbox("Select LGA", lga_ta['LGA'])
    additional_teachers = st.sidebar.number_input("Enter number of additional teachers", min_value=0, step=1)
    additional_students = st.sidebar.number_input("Enter number of additional students", min_value=0, step=1)
    
    # Display data
    st.subheader("LGA Data")
    st.write(lga_ta)


    # Calculate coverage
    if st.sidebar.button("Calculate Coverage"):
        percentage_coverage = calculate_coverage(lga_ta, lga, additional_teachers, additional_students)
        st.write(f"##### The new percentage coverage for {lga} is {percentage_coverage}%")

    st.divider()

    # Visualize the students and teachers distribution
    st.subheader("Data Visualization")

    st.write("#### SSS Students' Distribution Accross the 18 LGAs")
    st.bar_chart(lga_ta.set_index('LGA')['total_students'])

    st.write("#### SSS Teachers' Distribution Accross the 18 LGAs")
    st.bar_chart(lga_ta.set_index('LGA')['total_teachers'])

elif choose == "About":
    pass

elif choose == "Contact":
    pass


# the footer and more information
st.divider()
st.markdown(
    """<p style="color:white ; text-align:center;font-size:15px;"> Copyright | DSNai 2024(c) </p>
    """,
    unsafe_allow_html=True,
)

