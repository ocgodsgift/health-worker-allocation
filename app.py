import streamlit as st
from streamlit_option_menu import option_menu
from style_css import style
import pandas as pd
import numpy as np
from slideshow import slideshow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from datetime import datetime as t
from datetime import time


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
        ["Home", "Health", "Education", "Agriculture", "About", "Contact"],
        icons=["house", "heart", "book", "tree", "pen", "phone"],
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
        # df['Coverage'] = df.apply(lambda row: calculate_coverage(row['Doctors'], row['Nurses'], row['Population']), axis=1)

        for row in df.index:
            df.at[row,'Coverage'] = calculate_coverage(df.at[row,'Doctors'], df.at[row,'Nurses'], df.at[row,'Population'])

        # Streamlit UI
        # Sidebar state-wide adjustment inputs
        st.sidebar.header('State Health Workers Additions')
        state_additional_doctors = st.sidebar.number_input('Additional Doctors (State-wide)', min_value=0, value=0)
        state_additional_nurses = st.sidebar.number_input('Additional Nurses (State-wide)', min_value=0, value=0)

        # Add a button to trigger the calculation
        if st.sidebar.button('Calculate New Coverage'):
            # Distribute additional workforce proportionally based on population
            df['Additional Doctors'] = np.round((df['Population'] / df['Population'].sum() * state_additional_doctors),0).astype(int)
            df['Additional Nurses'] = np.round((df['Population'] / df['Population'].sum() * state_additional_nurses),0).astype(int)
            
            # Calculate new coverage with proposed additions
            df['New Doctors'] = df['Doctors'] + df['Additional Doctors']
            df['New Nurses'] = df['Nurses'] + df['Additional Nurses']

            for row in df.index:
                 df.at[row,'New Coverage'] = calculate_coverage(df.at[row,'New Doctors'], df.at[row,'New Nurses'], df.at[row,'Population'])
            
            # df['New Coverage'] = df.apply(lambda row: calculate_coverage(row['New Doctors'], row['New Nurses'], row['Population']), axis=1)

            # State-wide calculations
            total_population = df['Population'].sum()
            total_doctors = df['New Doctors'].sum()
            total_nurses = df['New Nurses'].sum()
            state_coverage = calculate_coverage(total_doctors, total_nurses, total_population)
            state_status = 'Meets WHO Standard' if state_coverage >= WHO_STANDARD else 'Below WHO Standard'

            def color_status(val):
                color = 'green' if val == 'Meets WHO Standard' else 'red'
                return f'color: {color}'

            # Display results
            st.write('#### State Health Workers Coverage with Proposed Additions')
            state_data = {
                'Total Population': [total_population],
                'Total Doctors': [total_doctors],
                'Total Nurses': [total_nurses],
                'State-Wide Coverage': [state_coverage],
                'Status': [state_status]
            }
            state_df = pd.DataFrame(state_data)
            st.dataframe(state_df.set_index('Total Population').style.applymap(color_status, subset=['Status']))

            # Display state-wide coverage status
            st.write('#### State-Wide Coverage with Proposed Additions')
            if state_coverage > 100:
                st.error(f"The new coverage of {state_coverage:.2f}% exceeds 100%. Please check the input values.")
            elif state_coverage < WHO_STANDARD:
                st.error(f"The new coverage of {state_coverage:.2f}% is below the WHO standard of {WHO_STANDARD}%.")
            else:
                st.success(f"The new coverage of {state_coverage:.2f}% meets the WHO standard of {WHO_STANDARD}%.")

            # Display health workforce data by LGA with proposed additions
            for row in df.index:
                 df.at[row,'Status'] = 'Meets WHO Standard' if df.at[row,'New Coverage'] >= WHO_STANDARD else 'Below WHO Standard'
                #  df['Status'] = df['New Coverage'].apply(lambda x: 'Meets WHO Standard' if x >= WHO_STANDARD else 'Below WHO Standard')


            st.write('#### Health Workforce Data by LGA with Additions')
            st.dataframe(df[['LGA', 'Population', 'Doctors', 'Nurses', 'Additional Doctors', 'Additional Nurses', 'New Doctors', 'New Nurses', 'New Coverage', 'Status']].set_index('LGA').style.applymap(color_status, subset=['Status']))

            # Graphs
            st.write('#### Visualizations')

            # Bar chart for doctors and nurses
            st.bar_chart(df.set_index('LGA')[['New Doctors', 'New Nurses']])

        else:
            total_population = df['Population'].sum()
            total_doctors = df['Doctors'].sum()
            total_nurses = df['Nurses'].sum()
            state_coverage = calculate_coverage(total_doctors, total_nurses, total_population)
            state_coverage = round(state_coverage, 2)
            state_status = 'Meets WHO Standard' if state_coverage >= WHO_STANDARD else 'Below WHO Standard'

            state_data = {
                'Total Population': [total_population],
                'Total Doctors': [total_doctors],
                'Total Nurses': [total_nurses],
                'State-wide Coverage': [state_coverage],
                'Status': [state_status]
            }

            state_df = pd.DataFrame(state_data)

            def color_status(val):
                color = 'green' if val == 'Meets WHO Standard' else 'red'
                return f'color: {color}'

            # Display initial health workforce state-wide
            st.write('#### State Health Workforce')
            st.dataframe(state_df.set_index('Total Population').style.applymap(color_status, subset=['Status']))

            # Display initial health workforce data by LGA
            for row in df.index:
                df.at[row,'Status'] = 'Meets WHO Standard' if df.at[row,'Coverage'] >= WHO_STANDARD else 'Below WHO Standard'

            st.write('#### Health Workforce Data by LGA')
            st.dataframe(df[['LGA', 'Population', 'Doctors', 'Nurses', 'Coverage','Status']].set_index('LGA').style.applymap(color_status, subset=['Status']))

        st.divider()

        st.write("""
            #### Overall Health Workers:
            To achieve good healthcare coverage, there should be at least 44.5 health workers (including doctors and nurses) for every 10,000 people.
            """)

    elif option == "OutPatient Scenario":

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
        st.write(lga_data.to_html(index=False), unsafe_allow_html=True)

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

    # Group by 'LGA' and calculate the sum of 'Total Students' and 'Total Teachers' for each 'LGA'
    lga_ta = ta_df.groupby('LGA').agg({'Total Students': 'sum', 'Total Teachers': 'sum'}).reset_index()

    # Calculate the ideal number of students per teacher
    ideal_students_per_teacher = 15

    # Calculate the actual number of students per teacher in each LGA
    lga_ta['Actual Students Per Teacher'] = round(lga_ta['Total Students'] / lga_ta['Total Teachers'], 2)

    # Calculate the percentage coverage in each LGA
    lga_ta['Percentage Coverage'] = round((ideal_students_per_teacher / lga_ta['Actual Students Per Teacher']) * 100, 2)

    lga_ta = lga_ta.sort_values(by='Percentage Coverage', ascending=True)

    # Function to calculate percentage coverage
    def calculate_coverage(df, lga, additional_teachers, additional_students, ideal_students_per_teacher):
        lga_row = df[df['LGA'] == lga]
        new_total_teachers = lga_row['Total Teachers'].values[0] + additional_teachers
        new_total_students = lga_row['Total Students'].values[0] + additional_students
        new_actual_students_per_teacher = new_total_students / new_total_teachers
        new_percentage_coverage = round((ideal_students_per_teacher / new_actual_students_per_teacher) * 100, 2)
        return new_total_students, new_total_teachers, new_percentage_coverage

    # Function to calculate the required additional teachers and students to achieve a desired percentage coverage
    def calculate_coverage_to_reach(df, lga, target_coverage, ideal_students_per_teacher):
        lga_row = df[df['LGA'] == lga]
        total_teachers = lga_row['Total Teachers'].values[0]
        total_students = lga_row['Total Students'].values[0]
        
        # Calculate the required actual students per teacher to achieve the target coverage
        required_actual_students_per_teacher = ideal_students_per_teacher / (target_coverage / 100)
        
        # Calculate the required total number of teachers to achieve the target coverage
        required_total_teachers = total_students / required_actual_students_per_teacher
        
        # Calculate the additional teachers needed
        additional_teachers = required_total_teachers - total_teachers
        
        return total_students, required_total_teachers, additional_teachers, required_actual_students_per_teacher

    # Streamlit interface
    st.title("Teachers Allocation Scenario Analysis")
    st.write("Welcome to the Teachers Allocation Scenario Analysis application!") 
    st.write("This tool is designed to help education administrators and policymakers analyze and optimize the allocation of senior secondary school (SSS) teachers across the 18 Local Government Areas (LGAs) in Edo State.")

    # Sidebar user inputs
    st.sidebar.header("Input Parameters")
    lga = st.sidebar.selectbox("Select LGA", lga_ta['LGA'])
    ideal_students_per_teacher = st.sidebar.number_input("Enter number of ideal students per teacher", min_value=15, step=5)
    additional_teachers = st.sidebar.number_input("Enter number of additional teachers", min_value=0, step=1)
    additional_students = st.sidebar.number_input("Enter number of additional students", min_value=0, step=1)


    # Calculate coverage and display results in a new table
    if st.sidebar.button("Calculate Coverage"):
        new_total_students, new_total_teachers, new_percentage_coverage = calculate_coverage(
            lga_ta, lga, additional_teachers, additional_students, ideal_students_per_teacher
        )
        new_data = {
            "LGA": [lga],
            "Additional Teachers": [additional_teachers],
            "Additional Students": [additional_students],
            "New Total Students": [new_total_students],
            "New Total Teachers": [new_total_teachers],
            "New Percentage Coverage": [new_percentage_coverage]
        }
        new_df = pd.DataFrame(new_data)
        st.subheader("New Coverage Data")
        st.write(new_df.to_html(index=False), unsafe_allow_html=True)
        st.write(f"The new percentage coverage for {lga} is {new_percentage_coverage}%")

    st.sidebar.subheader("Optimal Coverage")
    target_coverage = st.sidebar.slider("Select Target Coverage", 0, 100, 0)

    # Calculate coverage and display results in a new table
    if target_coverage > 0:
        total_students, required_total_teachers, additional_teachers, required_actual_students_per_teacher = calculate_coverage_to_reach(
            lga_ta, lga, target_coverage, ideal_students_per_teacher
        )
        new_data = {
            "LGA": [lga],
            "Total Students": [total_students],
            "Current Total Teachers": [lga_ta[lga_ta['LGA'] == lga]['Total Teachers'].values[0]],
            "Required Total Teachers": [round(required_total_teachers, 2)],
            "Additional Teachers Needed": [round(additional_teachers, 2)],
            "Target Percentage Coverage": [target_coverage]
        }
        new_df = pd.DataFrame(new_data)
        st.subheader("New Coverage Data")
        st.write(new_df.to_html(index=False), unsafe_allow_html=True)
        st.write(f"The new percentage coverage target for {lga} is {target_coverage}%")

    # Display original data
    st.subheader("Current Coverage Data")
    st.write(lga_ta.to_html(index=False), unsafe_allow_html=True)

elif choose == "Agriculture":

    # Importing tables
    table1 = pd.read_excel("Farmers Poverty Classified data.xlsx")
    loan = pd.read_excel("FARMERS WITH CROPS CLEANED UPDATED V1.xlsx", sheet_name = "LOAN")
    farm_type = pd.read_excel("FARMERS WITH CROPS CLEANED UPDATED V1.xlsx", sheet_name = 'farm_type')

    # Adding the information from 'Other - Educational Level', to the 'Educational Level' column
    cond = table1["Educational Level"] == 'Other'
    table1.loc[cond, 'Educational Level'] = table1.loc[cond, 'Other - Educational Level']

    # Merging the tables
    merg1 = table1.merge(loan, how = 'left', left_on = 'ID',right_on = 'ID')
    merged = merg1.merge(farm_type, how = 'left', left_on = 'ID',right_on = 'ID')

    def farmer_age(date):
        today = t.today
        age = today().year - date.year  - ((today().month, today().day) < (date.month, date.day))
        return age

    # grouping the age
    def age_group(df, age):
        bins = [0,30,50,120]
        labels = ['Young','Adult','Aged']
        df['Age group'] = pd.cut(df[age], bins = bins, labels = labels)
        return df


    def mappings(map_df):
        ed = {'Primary School':1,'Secondary School':2,'Other':0,'Tertiary':3,'Post Graduate':3,'Primary school':1,'Teachers Training':3,'Uneducated':0,'Technical college':2,'Informal education':1}
        bank = {'No':0,'Yes':1}
        ID = {'yes':1,'No':0}
        land = {'Rental':1,'Others':1,'Owner':2}
        sfarm = {'Large (6 Acre and above)':3,'Medium (1-6 Acre)':2,'Small (Less than 1 Acre)':1}
        mainc = {'Agriculture':1,'Others':0}
        agr = {'Crop':1,'Crop and Livestock':1,'Aquaculture':1,'Livestock':1,'Crop and Aquaculture':1,'Crop, Aquaculture and Livestock':1}
        coop = {'No':0,'Yes':1}
        noag = {"'4-7'":2,"'1-3'":1,"'8-11'":3}
        agem = {'Young':1,'Adult':3,'Aged':2}
        incom = {'0-100k':1,'100k-1m':2,'1m-10m':3,'10m-100m':4,'100m-500m':5,'500m and above':6}
        loan = {'yes':1}
        food ={'yes':1}
        cash = {'yes':1}
        Aqu = {'yes':1}
        Ls = {'yes':1}
        
        var = ['Educational Level','Do you have a bank account?','Identification','Type of Land Tenure',
        'Size of Farm','Major Source of Income(Agriculture)',"Agriculture Activity Engaged",
        'Are you in a cooperative?',"No. of Agricultural Activity Group","Age group","Income range",'loan',
        'Food','Cash','Aquatic','Livestock']

        map_ = [ed,bank,ID,land,sfarm,mainc,agr,coop,noag,agem,incom,loan,food,cash,Aqu,Ls]
        for i,j in zip(var,map_):
            map_df.loc[:,i] = map_df[i].map(j).fillna(0)
        return map_df

    #adding weights to the variables
    def weights(df):
        #Financial, Bank and Loan info
        df[["Income range","Do you have a bank account?","Identification","loan"]] = df[["Income range","Do you have a bank account?","Identification","loan"]] * 4

        #Farm land Information
        df[["Type of Land Tenure","Size of Farm"]] = df[["Type of Land Tenure","Size of Farm"]] * 3

        #Agricultural Activity
        df[["Major Source of Income(Agriculture)","Agriculture Activity Engaged","Food",
                "Cash","Aquatic","Livestock"]] = df[["Major Source of Income(Agriculture)","Agriculture Activity Engaged",
                                                        "Food","Cash","Aquatic","Livestock"]] * 2
        return df


    var = ['Educational Level','Do you have a bank account?','Identification','Type of Land Tenure',
        'Size of Farm','Major Source of Income(Agriculture)',"Agriculture Activity Engaged",
        'Are you in a cooperative?',"No. of Agricultural Activity Group","Age group","Income range",'loan',
        'Food','Cash','Aquatic','Livestock']

    # Drop all rows where the 'Date of Birth' and the 'Average Annual Income' empty.
    merged.dropna(subset = ["Date of Birth","Average Annual Income"], inplace = True)

    # Drop rows with bad date type
    row = merged['Date of Birth'].apply(lambda x: isinstance(x, time))
    merged = merged[~row]

    # Extracting the ages from the dtae column
    merged["Age"] = merged["Date of Birth"].apply(lambda x: farmer_age(x) if pd.notnull(x) else None)

    # Creating age group column
    merged = age_group(merged, 'Age')
    #merged["Age group"] = merged['Age'].apply(age_group)

    # change the data types to strings
    merged['No. of Agricultural Activity Group'] = merged['No. of Agricultural Activity Group'].astype('str')
    merged['Age group'] = merged['Age group'].astype('str')

    farmers = merged[var]
    farmers = mappings(farmers)
    farmers = weights(farmers)

    # Summing up all the the rows to create a credit score column
    farmers["credit_score"] = farmers[:].sum(axis = 1)

    # selecting the target and the independent variables.
    X = farmers.drop(["credit_score"], axis = 1)
    y = farmers.credit_score

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #print(f"Mean Squared Error: {mse}")
    #print(f"R^2 Score: {r2}")
    edu_level = ['Pre-primary','Primary School','Secondary School','Teachers Training','Tertiary','Post Graduate','Technical college','Informal education']
    bank_acc = ['No','Yes']
    ID = ['Yes','No']
    land_ten = ['Rental', 'Others', 'Owner']
    farm_size = ['Large (6 Acre and above)', 'Medium (1-6 Acre)','Small (Less than 1 Acre)']
    agric_incom_source = ['Agriculture', 'Others']
    agric_activity = ['Crop', 'Crop and Livestock', 'Aquaculture', 'Livestock','Crop and Aquaculture', 'Crop, Aquaculture and Livestock','Crop ']
    cooperative = ['No', 'Yes']
    No_agric_activity = ["'4-7'","'1-3'","'8-11'"]
    Incom = ['100k-1m', '0-100k','1m-10m', '10m-100m', '100m-500m','500m and above']
    loan = ['yes','no']
    Food = ['yes','no']
    Cash = ['yes','no']
    Aquatic = ['yes','no']
    Livestock = ['yes','no']


    st.title("Farmer Credit Scoring System")


    with st.form(key='my_form'):
        ag = st.number_input("How old are you?", value = 0)
        edu = st.selectbox("Educational Level", edu_level)
        bank = st.selectbox("Do you have a bank account?", bank_acc)
        id_ = st.selectbox("Do you have a mode of Identification (ID card)?", ID)
        ten = st.selectbox("Do you own your farmland?", land_ten)
        farmsize = st.selectbox("What is the size of your farm land?", farm_size)
        incsource = st.selectbox("What is your major source of income?", agric_incom_source)
        agactivity = st.selectbox("What king Agriculture do you practice?", agric_activity)
        coop = st.selectbox("Are you a registered member of any Agricultural cooperative?", cooperative)
        no_agractivity = st.selectbox("How many crop types/livestock types/Aquitic animal type do you farm?", No_agric_activity)
        income = st.selectbox("What is your annual income range?", Incom)
        lo = st.selectbox("Have you ever gotten a bank loan?", loan)
        foo = st.selectbox("Do you farm food crops", Food)
        cas = st.selectbox("Do you farm cash crops", Cash)
        aqu = st.selectbox("Do you do aquatic farming", Aquatic)
        livest = st.selectbox("Do you do livestock farming", Livestock)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:

        # Creating a new form
        form = {'Educational Level':[edu],'Do you have a bank account?':[bank],'Identification':[id_],
                'Type of Land Tenure':[ten],'Size of Farm':[farmsize],
                'Major Source of Income(Agriculture)':[incsource],
                "Agriculture Activity Engaged":[agactivity],'Are you in a cooperative?':[coop],
                "No. of Agricultural Activity Group":[no_agractivity],"Age group":[ag],"Income range":[income],
                'loan':[lo],'Food':[foo],'Cash':[cas],'Aquatic':[aqu],'Livestock':[livest]}

        # Applying the processing functions on the new form
        form = pd.DataFrame(form)
        form = age_group(form,'Age group')
        form['Age group'] = form['Age group'].astype('str')
        form = mappings(form)

        pred = model.predict(form)
        st.write(f'Your Credit score is {pred[0]}')


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

