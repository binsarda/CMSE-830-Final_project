import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from PIL import Image
import hiplot as hip
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,f1_score
from sklearn.model_selection import cross_val_score



stroke=pd.read_csv("stroke.csv")

stroke["hypertension"].replace([1,0],["Yes","No"],inplace=True) # already done in main dataset
stroke["heart_disease"].replace([1,0],["Yes","No"],inplace=True) # already done in main dataset
stroke["stroke"].replace([1,0],["Yes","No"],inplace=True)

col1, col2, col3 = st.columns([1, 8, 1])
col2.markdown("### :blue[Please scroll to right to explore more tabs]")
tab_titles=[
    "Home",
    "EDA",
    "EDA(RC)",
    "Reg.(T1)",
    "Reg.(T2)",
    "Reg.(T3)",
    "PCA",
    "Classification",
    "Discussion"
]

tabs=st.tabs(tab_titles)
with tabs[0]:
    st.markdown("# !!!! Welcome to my website !!!!")
    col1, col2, col3 = st.columns([1, 2, 1])

    logo = Image.open('msu_logo.png')
    col1.image(logo, caption='MSU')

    egr = Image.open('msu.png')
    egr = egr.resize((1000, 1000))
    col3.image(egr, caption='Spartans')

    cmse = Image.open('cmse.jfif')


    button_home_sidbar_status = True

    button_home_sidbar = col1.button(":red[Show Github Link]")


    if col3.button(":red[Hide Github Link]"):
        button_home_sidbar_status= False
        button_home_sidbar = False

    if button_home_sidbar_status | button_home_sidbar:
        st.sidebar.write("My github link: [link](https://github.com/binsarda?tab=repositories)")
        st.sidebar.image(cmse, caption='Data Science and Machine Learning')





    col1, col2, col3 = st.columns([1, 2, 1])
    col1.write("Owner Name: Sardar Nafis Bin Ali")
    col3.write("Institution: Michigan State University")
    st.markdown("# :red[Click below to know about the author of this site!]")
    with st.expander(label=("# :violet[Click here to know about the author of this site!]"),expanded=False):
        nafispic = Image.open('nafis.jpg')
        st.image(nafispic, caption='Sardar Nafis Bin Ali')
        st.write("Sardar is driven by a thirst for knowledge and has a spirit of exploration. "
                 "He focused on the captivating domain of high-speed aerodynamics during his undergraduate "
                 "studies in mechanical engineering. Currently, he is pursuing "
                 "a Ph.D. in mechanical engineering and planning to do a dual"
                 " degree with the department of communicative sciences and "
                 "disorders. He aims to learn about the intricate aspects of"
                 "human communication and contribute to advancements in voice science."
                 " Apart from his academic pursuits, he finds solace in traveling, "
                 "embracing diverse cultures, and capturing the world's beauty through "
                 "his experiences. Sardar actively engages in initiatives "
                 "promoting sustainability and environmental conservation. "
                 "With an unquenchable curiosity and unwavering dedication, "
                 "he continues to make a meaningful impact in his chosen fields and beyond.")

    button = st.radio('Do you want to delete any row having NaN in at least one of the fields', ['Yes', 'No'])
    if button == 'Yes':
        df = stroke.dropna()
        st.write("You deleted rows having NaN in at least one of the fields")
    elif button == 'No':
        df = stroke
    st.write(df)

    st.markdown("# :red[Click below to learn about ' Brain Stroke' dataset.]")
    with st.expander("# :violet[Click here to learn about ' Brain Stroke' dataset. ]"):
        strokepic = Image.open('strokepic.jpg')
        st.image(strokepic, caption='Brain-Stroke')
        st.write("Finding significant insights and patterns within"
                 " the variables of this 'Brain Stroke' dataset is the main objective of "
                 "data visualization in the context of exploratory data "
                 "analysis (EDA). We want to learn more about the potential "
                 "connections between the prevalence of strokes and various "
                 "variables like age, gender, particular health conditions, lifestyle"
                 " preferences, and occupation. Our goal is to lay the groundwork "
                 "for deeper analyses and the creation of predictive models, ultimately "
                 "facilitating a more thorough assessment of stroke risk. We do this by "
                 "using visual representations that highlight distributions, associations, and correlations.")

    col1, col2, col3 = st.columns([2, 1, 2])
    button1 = col1.button(":red[Show Statistics]")
    if button1:
        st.sidebar.write(df.describe())

    if col3.button(":red[Hide Statistics]"):
        button1 = False

    cols = df.columns
    numcols = df.select_dtypes(include=[np.number]).columns
    strcols = df.select_dtypes(include=['object']).columns
    numcoldf = df[numcols]
    strcoldf = df[strcols]

    col1, col2, col3 = st.columns([2, 1, 2])
    button2 = col1.button(":red[Show Columns]")
    if button2:
        st.sidebar.write("No. of columns are ", len(cols))
        st.sidebar.write("The columns are following-")
        st.sidebar.write(df.columns)
        st.sidebar.write("Name of columns containing numerical values: ")
        st.sidebar.write(numcols)
        st.sidebar.write("Name of columns containing string or non-numerical values: ")
        st.sidebar.write(strcols)
    if col3.button(":red[Hide Columns]"):
        button2 = False


with tabs[1]: # EDA page
    st.markdown("# Exploratory Data Analysis(EDA)")
    col1, col2, col3 = st.columns([2, 2, 3])
    col3.header(" :red[Interactive Plot]")
    exp = hip.Experiment.from_dataframe(df)
    htmlcomp = exp.to_html()
    st.components.v1.html(htmlcomp, width=1000, height=700, scrolling=True)

    st.header("Different kinds of plots")
    st.markdown("# :red[Please select kind of visualization you want to see[from below] and also select proper variable from the options below...]")

    graph_button = st.selectbox('Please select one kind of graph from the following options:',
                                [ 'HeatMap', 'Bar Plot', 'Violin Plot', 'Box Plot', '2-D Scatter Plot',
                                 '3-D Scatter Plot'])
    if graph_button == 'HeatMap':
        sns.heatmap(numcoldf.corr(), annot=True)
        st.pyplot(plt.gcf())

    elif graph_button == 'Bar Plot':
        st.write("Please select following variables for bar plot")
        xv = st.selectbox('Please select x or first variable for bar plot:', cols)
        yv = st.selectbox('Please select y or second variiable for bar plot:', cols)
        st.bar_chart(data=df, x=xv, y=yv)
        st.pyplot(plt.gcf())

    elif graph_button == 'Violin Plot':
        st.write("Please select following variables for violin plot")
        xv = st.selectbox('Please select x or first variable for violin plot:', strcols)
        yv = st.selectbox('Please select y or second variiable for violin plot:', numcols)
        zv = st.selectbox('Please select hue or third variiable for violin plot:', strcols)
        sns.violinplot(data=df, x=xv, y=yv, hue=zv)
        st.pyplot(plt.gcf())




    elif graph_button == 'Box Plot':
        st.write("Please select following variables for  plot")
        xv = st.selectbox('Please select x or first variable for  plot:', strcols)
        yv = st.selectbox('Please select y or second variiable for  plot:', numcols)
        zv = st.selectbox('Please select hue or third variiable for   plot:', strcols)
        sns.boxplot(x=xv, y=yv, hue=zv, data=df)
        st.pyplot(plt.gcf())

    elif graph_button == '2-D Scatter Plot':
        st.write("Please select following variables for  plot")
        xv = st.selectbox('Please select x or first variable for  plot:', numcols)
        yv = st.selectbox('Please select y or second variiable for  plot:', numcols)
        zv = st.selectbox('Please select z or hue or third variiable for  plot:', strcols)

        a = sns.scatterplot(data=df, x=xv, y=yv, hue=zv)
        st.pyplot(plt.gcf())
    elif graph_button == '3-D Scatter Plot':
        st.write("Please select following variables for  plot")
        xv = st.selectbox('Please select x or first variable for 3-d plot:', numcols)
        yv = st.selectbox('Please select y or second variiable for 3-d plot:', numcols)
        zv = st.selectbox('Please select z or or third variiable for  3-d plot:', numcols)


        fig = px.scatter_3d(x=df[xv], y=df[yv], z=df[zv], color_discrete_sequence=['green'])
        fig.update_layout(scene=dict(xaxis_title=xv, yaxis_title=yv, zaxis_title=zv), title="3-d Scatter Plot")
        st.plotly_chart(fig)


with tabs[2]: # EDA with reduced dataset
    st.markdown("# Exploratory Data Analysis(EDA) with Reduced Dataset")
    st.header(
        "Please select reduced number of columns for Reduced Dataset (Select at least 3 variables (at least 2  of numerical type"
        " and at least one  of string or non-numerical type))")
    red_cols = st.multiselect('Pick the columns', cols)

    if len(red_cols) > 0:
        red_df = df[red_cols]
        st.write(
            f"You have choosen {len(red_cols)} number of columns in datatset and number of different column is {len(red_cols)} ")
        st.write("Reduced Dataset")
        st.write(red_df.head(10))
        red_numcols = red_df.select_dtypes(include=[np.number]).columns
        red_strcols = red_df.select_dtypes(include=['object']).columns
        red_ndf = df[red_numcols]
        red_sdf = df[red_strcols]
        st.sidebar.write("For reduced dataset")
        st.sidebar.write("No. of columns are ", len(red_cols))
        st.sidebar.write("The columns are following-")
        st.sidebar.write(red_df.columns)
        st.sidebar.write("Name of columns containing numerical values: ")
        st.sidebar.write(red_numcols)
        st.sidebar.write("Name of columns containing string or non-numerical values: ")
        st.sidebar.write(red_strcols)
        if len(red_numcols) == 1:
            st.write("Please select following variables for different plotting (for reduced dataset)")
            rxv = st.selectbox('(For reduced dataset) Please select x or first variable:', red_numcols)

        if len(red_numcols) >= 2:
            st.write("Please select following variables for different plotting (for reduced dataset)")
            rxv = st.selectbox('(For reduced dataset) Please select x or first variable:', red_numcols)
            ryv = st.selectbox('(For reduced dataset) Please select y or second variiable:', red_numcols)
            if len(red_strcols) >= 1:
                rzv = st.selectbox('(For reduced dataset) Please select hue or third variiable:', red_strcols)

            plot1 = plt.figure(figsize=(10, 4))
            sns.lineplot(x=rxv, y=ryv, data=red_df)
            st.pyplot(plot1)

            plot2 = sns.pairplot(red_df)
            st.pyplot(plot2.fig)

            plot3 = sns.heatmap(red_ndf.corr(), annot=True)
            st.pyplot(plot3.get_figure())

            fig4, ax4 = plt.subplots()
            sns.heatmap(red_ndf.corr(), ax=ax4, annot=True)
            st.write(fig4)



with tabs[3]: # Regression without train test
    st.markdown("# Different kinds of curve fitting between 2 variables")

    st.markdown("## :blue[Equation of the fitted curve is: ]")
    st.markdown("## :violet[$y=w_0+w_1x+w_2x^2+w_3x^3+..........+w_nx^n$]")
    st.markdown("## :blue[n is the degree of polynomial]")

    st.markdown("## :red[Select 2 variables below:]")
    x_col1 = st.selectbox(' Please select x or independent variable for linear regression:', ["age","avg_glucose_level","bmi"] )
    y_col1 = st.selectbox(' Please select y or dependent variable for linear regression:', ["bmi","avg_glucose_level","age"])
    x = df[x_col1]
    y = df[y_col1]

    x = np.array(x)
    y = np.array(y)

    degree =  st.number_input("Insert the degree of curve to be fitted", value=1, placeholder="Type a number...",min_value=1, max_value=25, step=1)
    mat = np.zeros((x.shape[0], degree + 1))
    for i in range(x.shape[0]):
        for j in range(degree + 1):
            if j == 0:
                mat[i, j] = 1
            else:
                mat[i, j] = x[i] ** j
    co_effs = np.linalg.pinv(mat) @ y.T

    y_pred = np.zeros(x.shape[0])
    for i in range(co_effs.shape[0]):
        y_pred = y_pred + co_effs[i] * (x ** i)

    R_sq = sum((abs(y - y_pred)) ** 2)
    R_sq = R_sq / x.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"### :violet[ R value is {round(R,4)}]")

    plt.clf()
    plt.scatter(x, y, label="Scatter plot of main data")
    plt.plot(x, y_pred, 'ro', label="Regression Curve")
    plt.title("Scatter and Regression")
    plt.xlabel(x_col1)
    plt.ylabel(y_col1)
    plt.legend()
    st.pyplot(plt)

    st.markdown(f"## :green[Prediction!!!] ")
    xpt1=st.number_input(f"Insert value of {x_col1} to predict your {y_col1}", value=0)
    ypt1 = 0
    for i in range(co_effs.shape[0]):
        ypt1 = ypt1 + co_effs[i] * (xpt1 ** i)
    st.markdown(f"## Your predicted {y_col1} value is {round(ypt1,4)} based on your {x_col1}={xpt1}. ")


    st.markdown("## :red[Curve fitting with sinusoidals]")
    st.markdown("## :blue[Equation of the following fitted curve is: ]")
    st.markdown("## :violet[$ w_0+w_1x+w_2Sin(x)+w_3Cos(x) $]")

    mat = np.zeros((x.shape[0], 4))
    for i in range(x.shape[0]):
        for j in range(4):
            if j == 0:
                mat[i, j] = 1
            elif j == 1:
                mat[i, j] = x[i]
            elif j == 2:
                mat[i, j] = np.sin((180 / np.pi) * x[i])
            elif j == 2:
                mat[i, j] = np.cos((180 / np.pi) * x[i])

    co_effs = np.linalg.pinv(mat) @ y.T

    y_pred = np.zeros(x.shape[0])

    y_pred = co_effs[0] * np.ones(x.shape[0]) + co_effs[1] * x + co_effs[2] * np.sin((180 / np.pi) * x) + co_effs[
        3] * np.cos((180 / np.pi) * x)

    R_sq = sum((abs(y - y_pred)) ** 2)
    R_sq = R_sq / x.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"### :violet[R value is {round(R,4)}]")

    plt.clf()
    plt.scatter(x, y, label="Scatter plot of main data")
    plt.plot(x, y_pred, 'ro', label="Regression Curve")
    plt.title("Scatter and Regression")
    plt.xlabel(x_col1)
    plt.ylabel(y_col1)
    plt.legend()
    st.pyplot(plt)


    st.markdown(f"## :green[Prediction!!!] ")
    xpt2 = st.number_input(f"Insert value of {x_col1} to predict your {y_col1} with sinusoidal fitting..", value=0)
    ypt2 = 0

    ypt2 = co_effs[0]  + co_effs[1] * xpt2 + co_effs[2] * np.sin((180 / np.pi) * xpt2) + co_effs[
        3] * np.cos((180 / np.pi) * xpt2)
    st.markdown(f"## Your predicted {y_col1} value is {round(ypt2, 4)} based on your {x_col1}={xpt2}. ")




with tabs[4]:# Regression with test and training and testing data
    st.markdown("## :blue[Here we will do regression between 2 variables considering training and test sets]")

    x_col2 = st.selectbox(' Please select x or independent variable for the linear regression:', ["age","avg_glucose_level","bmi"] )
    y_col2 = st.selectbox(' Please select y or dependent variable for the linear regression:',[ "bmi", "avg_glucose_level","age"])

    x = df[x_col2]
    y = df[y_col2]


    x = np.array(x)
    y = np.array(y)

    total_num = x.shape[0]
    test_p1 = st.number_input("Insert the (percentage) of test samples", value=15, placeholder="Type a number.....",min_value=10, max_value=40, step=1)
    test_p = test_p1 / 100
    test_num = round(total_num * test_p)
    train_num = total_num - test_num



    total_arr = np.arange(total_num)
    test_arr = np.array(random.sample(list(total_arr), test_num))
    train_arr = np.setxor1d(total_arr, test_arr)

    x_train = np.zeros(train_num)
    y_train = np.zeros(train_num)
    x_test = np.zeros(test_num)
    y_test = np.zeros(test_num)

    for i in range(train_num):
        x_train[i] = x[train_arr[i]]
        y_train[i] = y[train_arr[i]]
    for i in range(test_num):
        x_test[i] = x[test_arr[i]]
        y_test[i] = y[test_arr[i]]



    # For train set only

    degree = st.number_input("Insert the degree of polynomial to be fitted for this.. ", value=1, placeholder="Type a number...",min_value=1, max_value=25, step=1)
    mat = np.zeros((x_train.shape[0], degree + 1))
    for i in range(x_train.shape[0]):
        for j in range(degree + 1):
            if j == 0:
                mat[i, j] = 1
            else:
                mat[i, j] = x_train[i] ** j
    co_effs = np.linalg.pinv(mat) @ y_train.T

    y_pred_train = np.zeros(x_train.shape[0])
    for i in range(co_effs.shape[0]):
        y_pred_train = y_pred_train + co_effs[i] * (x_train ** i)

    R_sq = sum((abs(y_train - y_pred_train)) ** 2)
    R_sq = R_sq / x_train.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"## :violet[For Training set only]")
    st.markdown(f"## :violet[R value is {R}]")

    plt.clf()
    plt.scatter(x_train, y_train, label="Scatter plot of main Training data")
    plt.plot(x_train, y_pred_train, 'go', label="Regression Curve")
    plt.title("Scatter and Regression for Training data")
    plt.xlabel(x_col2)
    plt.ylabel(y_col2)
    plt.legend()
    st.pyplot(plt)


    #For test set only

    y_pred_test = np.zeros(x_test.shape[0])
    for i in range(co_effs.shape[0]):
        y_pred_test = y_pred_test + co_effs[i] * (x_test ** i)

    R_sq = sum((abs(y_test - y_pred_test)) ** 2)
    R_sq = R_sq / x_test.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"## :violet[For Test set only]")
    st.markdown(f"## :violet[R value is {R}]")

    plt.clf()
    plt.scatter(x_test, y_test, label="Scatter plot of main Test data")
    plt.plot(x_test, y_pred_test, 'go', label="Regression Curve")
    plt.title("Scatter and Regression for Test data")
    plt.xlabel(x_col2)
    plt.ylabel(y_col2)
    plt.legend()
    st.pyplot(plt)


    #Entire data(training+testing set)
    y_pred = np.zeros(x.shape[0])
    for i in range(co_effs.shape[0]):
        y_pred = y_pred + co_effs[i] * (x ** i)

    R_sq = sum((abs(y - y_pred)) ** 2)
    R_sq = R_sq / x.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"## :violet[For Entire set ]")
    st.markdown(f"## :violet[R value is {R}]")

    plt.clf()
    plt.scatter(x, y, label="Scatter plot of main  data")
    plt.plot(x, y_pred, 'go', label="Regression Curve")
    plt.title("Scatter and Regression for Entire data")
    plt.xlabel(x_col2)
    plt.ylabel(y_col2)
    plt.legend()
    st.pyplot(plt)

    st.markdown(f"## :green[Prediction!!!] ")
    xpt3 = st.number_input(f"Insert value of {x_col2} for predicting your {y_col2} with this..", value=0)
    ypt3 = 0

    for i in range(co_effs.shape[0]):
        ypt3 = ypt3 + co_effs[i] * (xpt3 ** i)
    st.markdown(f"## Your predicted {y_col2} value is {round(ypt3, 4)} based on your {x_col2}={xpt3}. ")






with tabs[5]:# Multiple  Regression
    st.markdown("## :blue[Multiple variable Regression]")
    st.markdown("## :blue[Equation of the fitted curve is: ]")
    st.markdown("## :violet[$y=w_0+w_1x_1^2+w_2x_2^2+w_3x_1+w_4x_2+w_5x_1x_2$]")
    x_col1_m = st.selectbox(' Please select x1 or 1st independent variable for the multiple variable regression:', ["age","avg_glucose_level","bmi"] )
    x_col2_m = st.selectbox(' Please select x2 or 2nd independent variable for the multiple  variable regression:', ["avg_glucose_level","bmi","age"] )
    y_col_m = st.selectbox(' Please select y or dependent variable for the multiple variable regression:', ["bmi","age","avg_glucose_level"] )

    x1 = df[x_col1_m]
    x2 = df[x_col2_m]
    y = df[y_col_m]

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    mat = np.zeros((x.shape[0], 6))
    for i in range(x.shape[0]):
        for j in range(6):
            if j == 0:
                mat[i, j] = 1
            elif j == 1:
                mat[i, j] = x1[i] ** 2
            elif j == 2:
                mat[i, j] = x2[i] ** 2
            elif j == 3:
                mat[i, j] = x1[i]
            elif j == 4:
                mat[i, j] = x2[i]
            elif j == 5:
                mat[i, j] = x1[i] * x2[i]

    co_effs = np.linalg.pinv(mat) @ y.T

    y_pred = co_effs[0] * np.ones(x.shape[0]) + co_effs[1] * x1 ** 2 + co_effs[2] * x2 ** 2 + co_effs[3] * x1 + co_effs[
        4] * x2 + co_effs[5] * x1 * x2

    R_sq = sum((abs(y - y_pred)) ** 2)
    R_sq = R_sq / x.shape[0]
    R = np.sqrt(R_sq)
    st.markdown(f"## :violet[R value is {R}]")

    fig = px.scatter_3d(x=x1, y=x2, z=y)
    fig.update_layout(scene=dict(xaxis_title=x_col1_m, yaxis_title=x_col2_m, zaxis_title=y_col_m), title="Scatter plot of Original Data")
    st.plotly_chart(fig)

    fig = px.scatter_3d(x=x1, y=x2, z=y_pred, color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col1_m, yaxis_title=x_col2_m, zaxis_title=y_col_m), title="Hyper-plane from Regression")
    st.plotly_chart(fig)

    st.markdown(f"## :green[Prediction!!!] ")
    xp1_m = st.number_input(f"Insert value of {x_col1_m} for predicting your {y_col_m} with this method..", value=0)
    xp2_m = st.number_input(f"Insert value of {x_col2_m} for predicting your {y_col_m} with this method..", value=0)

    yp_m = 0

    yp_m = co_effs[0]  + co_effs[1] *  xp1_m  ** 2 + co_effs[2] *  xp2_m **2  + co_effs[3] *  xp1_m  + co_effs[4] *  xp2_m  + co_effs[5] *  xp1_m  *  xp2_m
    st.markdown(f"## Your predicted {y_col_m} value is {round(yp_m, 4)} based on your {x_col1_m}={xp1_m} and {x_col2_m}={xp2_m} . ")
















with tabs[6]:# PCA
    st.markdown("## :blue[Principal Component Analysis(PCA)]")

    x_col11 = st.selectbox(' Please select x1 or 1st independent variable for PCA:', ["age","avg_glucose_level","bmi"] )
    x_col22 = st.selectbox(' Please select x2 or 2nd independent variable for PCA:', ["bmi","age","avg_glucose_level"] )
    y_col111 = st.selectbox(' Please select y or dependent variable for PCA:', ["avg_glucose_level","bmi","age"] )

    x1=df[x_col11]
    x2=df[x_col22]
    y=df[y_col111]

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    mat = np.stack((x1, x2, y)).T
    U, sig, Vp = np.linalg.svd(mat)

    sig_full = np.zeros((x1.shape[0], 3))
    for i in range(x1.shape[0]):
        for j in range(3):
            if i == j:
                sig_full[i, j] = sig[i]

    sig_1 = sig_full.copy()
    sig_1[1, 1] = 0
    sig_1[2, 2] = 0

    sig_2 = sig_full.copy()
    sig_2[0, 0] = 0
    sig_2[2, 2] = 0

    sig_3 = sig_full.copy()
    sig_3[0, 0] = 0
    sig_3[1, 1] = 0

    sig_12 = sig_1 + sig_2

    recons_full = U @ sig_full @ Vp
    recons_1 = U @ sig_1 @ Vp
    recons_2 = U @ sig_2 @ Vp
    recons_3 = U @ sig_3 @ Vp
    recons_12 = U @ sig_12 @ Vp

    fig = px.scatter_3d(x=recons_full[:, 0], y=recons_full[:, 1], z=recons_full[:, 2], color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111),title="Data with all Principal values")
    st.plotly_chart(fig)

    fig = px.scatter_3d(x=recons_1[:, 0], y=recons_1[:, 1], z=recons_1[:, 2], color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111), title="Principal Axis 1")
    st.plotly_chart(fig)


    fig = px.scatter_3d(x=recons_2[:, 0], y=recons_2[:, 1], z=recons_2[:, 2], color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111), title="Principal Axis 2")
    st.plotly_chart(fig)


    fig = px.scatter_3d(x=recons_3[:, 0], y=recons_3[:, 1], z=recons_3[:, 2], color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111), title="Principal Axis 3")
    st.plotly_chart(fig)


    fig = px.scatter_3d(x=np.concatenate((recons_1[:, 0], recons_2[:, 0], recons_3[:, 0])),y=np.concatenate((recons_1[:, 1], recons_2[:, 1], recons_3[:, 1])),z=np.concatenate((recons_1[:, 2], recons_2[:, 2], recons_3[:, 2])),color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111), title="All Principal Axes")
    st.plotly_chart(fig)


    fig = px.scatter_3d(x=recons_12[:, 0], y=recons_12[:, 1], z=recons_12[:, 2], color_discrete_sequence=['green'])
    fig.update_layout(scene=dict(xaxis_title=x_col11, yaxis_title=x_col22, zaxis_title=y_col111), title="Data reconstruction with 2 highest Principal Values")
     #fig.show()
    st.plotly_chart(fig)


with tabs[7]:# Classification
    st.markdown("## :blue[Different Machine Learning Algorithms for Classifications]")
    # converting the classes into values
    df2 = df.copy()

    df2["gender"].replace(["Male", "Female"], [1, 0], inplace=True)

    df2["hypertension"].replace(["Yes", "No"], [1, 0], inplace=True)

    df2["heart_disease"].replace(["Yes", "No"], [1, 0], inplace=True)

    df2["ever_married"].replace(["Yes", "No"], [1, 0], inplace=True)

    df2["work_type"].replace(["Private", "Self-employed", "Govt_job", "children"], [0, 1, 2, 3], inplace=True)

    df2["Residence_type"].replace(["Urban", "Rural"], [0, 1], inplace=True)

    df2["smoking_status"].replace(["formerly smoked", "never smoked", "smokes", "Unknown"], [0, 1, 2, 3], inplace=True)

    df2["stroke"].replace(["Yes", "No"], [1, 0], inplace=True)

    st.markdown("## :violet[Select catagorical values from below]")
    gender_vi = st.selectbox(' Please select  Gender:', ["Male", "Female"])
    hypertension_vi = st.selectbox(' Please select  Hyper-tension status:', ["Yes", "No"])
    heart_disease_vi = st.selectbox(' Please select  Heart-disease status:', ["Yes", "No"])
    ever_married_vi = st.selectbox(' Please select  Marital status::', ["Yes", "No"])
    work_type_vi = st.selectbox(' Please select  Work-Type:', ["Private", "Self-employed", "Govt_job", "children"])
    residence_type_vi = st.selectbox(' Please select  Residence-Type:', ["Urban", "Rural"])
    smoking_status_vi = st.selectbox(' Please select  Smoking status:', ["formerly smoked", "never smoked", "smokes", "Unknown"])

    if gender_vi == "Male":
        gender_vo = 1
    elif gender_vi == "Female":
        gender_vo = 0

    # Hypertension
    if hypertension_vi == "Yes":
        hypertension_vo = 1
    elif hypertension_vi == "No":
        hypertension_vo = 0

    # Heart Disease
    if heart_disease_vi == "Yes":
        heart_disease_vo = 1
    elif heart_disease_vi == "No":
        heart_disease_vo = 0

    # Ever Married
    if ever_married_vi == "Yes":
        ever_married_vo = 1
    elif ever_married_vi == "No":
        ever_married_vo = 0

    # Work Type
    if work_type_vi == "Private":
        work_type_vo = 0
    elif work_type_vi == "Self-employed":
        work_type_vo = 1
    elif work_type_vi == "Govt_job":
        work_type_vo = 2
    elif work_type_vi == "children":
        work_type_vo = 3

    # Residence type
    if residence_type_vi == "Urban":
        residence_type_vo = 0
    elif residence_type_vi == "Rural":
        residence_type_vo = 1

    # Smoking Status
    if smoking_status_vi == "formerly smoked":
        smoking_status_vo = 0
    elif smoking_status_vi == "never smoked":
        smoking_status_vo = 1
    elif smoking_status_vi == "smokes":
        smoking_status_vo = 2
    elif smoking_status_vi == "Unknown":
        smoking_status_vo = 3

    st.markdown("## :violet[Select numerical values from below]")
    age_input = st.number_input("Insert the value of age", value=30, placeholder="Type a number..........", min_value=1,max_value=100, step=1)
    avg_glucose_level_input = st.number_input("Insert the value of Average Glucose Level", value=1500.00,placeholder="Type a number..........", min_value=0.00, max_value=10000.00,step=0.2)
    bmi_input = st.number_input("Insert the value of BMI", value=1500.00, placeholder="Type a number..........", min_value=0.00, max_value=10000.00, step=0.2)

    features = np.array(df2.iloc[:, 0:10])
    labels = np.array(df2['stroke'])

    st.markdown("## :violet[Insert the (percentage) of total data sample for test samples below]")
    test_fraction_per = st.number_input("Insert the (percentage) of total data sample for test samples", value=20, placeholder="Type a number..........", min_value=5, max_value=40, step=1)
    test_fraction = test_fraction_per / 100;
    x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(features, labels, test_size=test_fraction, random_state=0)

    st.markdown("## :violet[Insert the Classifier Model from below]")
    classifier_type = st.selectbox(' Please select a classifier Type:',["K Nearest Neighbors(KNN)", "Logistic Regression", "Support Vector Machine","Random Forest","Decision Tree"])



    if classifier_type=="K Nearest Neighbors(KNN)":
        modell = KNeighborsClassifier()
        neighbors_num=st.number_input("Specify number of neighbors to be considered", value=3, placeholder="Type a number..........", min_value=1,max_value=30, step=1)
        modell = KNeighborsClassifier(n_neighbors=neighbors_num)
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        st.markdown(f"## The score of the model used is :red[{round(sc,4)}]")



        cv_nums= st.number_input("Insert the value Cross Folds for validation", value=4, placeholder="Type a number..........", min_value=2,max_value=30, step=1)
        cv_sc = cross_val_score(modell, features, labels, cv=cv_nums)
        st.markdown("## The cross validation scores of the model used are ")
        for i in range(cv_nums):
            st.markdown(f"## :red[{round(cv_sc[i],4)}]")

        y_cat_pred = modell.predict(x_cat_test)
        st.markdown(f"## The f1 score of the model used is :red[{round(f1_score(y_cat_test, y_cat_pred, average='weighted'), 4)}]")

        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)


        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)


        # Streamlit app layout
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gender_vo,age_input,hypertension_vo,heart_disease_vo,ever_married_vo,work_type_vo,residence_type_vo,avg_glucose_level_input,bmi_input,smoking_status_vo]] ))
        if pred_res==0:
            pred_res="No"
        elif pred_res==1:
            pred_res="Yes"

        st.markdown(f"### Your predicted stroke status :blue[is:  {pred_res}] based on the input values.")




    if classifier_type=="Logistic Regression":

        modell = LogisticRegression()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        st.markdown(f"## The score of the model used is :red[{round(sc,4)}]")



        cv_nums= st.number_input("Insert the value Cross Folds for validation", value=4, placeholder="Type a number..........", min_value=2,max_value=30, step=1)
        cv_sc = cross_val_score(modell, features, labels, cv=cv_nums)
        st.markdown("## The cross validation scores of the model used are ")
        for i in range(cv_nums):
            st.markdown(f"## :red[{round(cv_sc[i],4)}]")

        y_cat_pred = modell.predict(x_cat_test)
        st.markdown(f"## The f1 score of the model used is :red[{round(f1_score(y_cat_test, y_cat_pred, average='weighted'), 4)}]")

        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)


        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)


        # Streamlit app layout
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gender_vo,age_input,hypertension_vo,heart_disease_vo,ever_married_vo,work_type_vo,residence_type_vo,avg_glucose_level_input,bmi_input,smoking_status_vo]] ))
        if pred_res==0:
            pred_res="No"
        elif pred_res==1:
            pred_res="Yes"

        st.markdown(f"### Your predicted stroke status :blue[is:  {pred_res}] based on the input values.")





    if classifier_type=="Support Vector Machine":

        modell = SVC()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        st.markdown(f"## The score of the model used is :red[{round(sc,4)}]")



        cv_nums= st.number_input("Insert the value Cross Folds for validation", value=4, placeholder="Type a number..........", min_value=2,max_value=30, step=1)
        cv_sc = cross_val_score(modell, features, labels, cv=cv_nums)
        st.markdown("## The cross validation scores of the model used are ")
        for i in range(cv_nums):
            st.markdown(f"## :red[{round(cv_sc[i],4)}]")

        y_cat_pred = modell.predict(x_cat_test)
        st.markdown(f"## The f1 score of the model used is :red[{round(f1_score(y_cat_test, y_cat_pred, average='weighted'), 4)}]")

        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)


        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)


        # Streamlit app layout
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gender_vo,age_input,hypertension_vo,heart_disease_vo,ever_married_vo,work_type_vo,residence_type_vo,avg_glucose_level_input,bmi_input,smoking_status_vo]] ))
        if pred_res==0:
            pred_res="No"
        elif pred_res==1:
            pred_res="Yes"

        st.markdown(f"### Your predicted stroke status :blue[is:  {pred_res}] based on the input values.")








    if classifier_type=="Random Forest":

        modell = RandomForestClassifier()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        st.markdown(f"## The score of the model used is :red[{round(sc,4)}]")



        cv_nums= st.number_input("Insert the value Cross Folds for validation", value=4, placeholder="Type a number..........", min_value=2,max_value=30, step=1)
        cv_sc = cross_val_score(modell, features, labels, cv=cv_nums)
        st.markdown("## The cross validation scores of the model used are ")
        for i in range(cv_nums):
            st.markdown(f"## :red[{round(cv_sc[i],4)}]")

        y_cat_pred = modell.predict(x_cat_test)
        st.markdown(f"## The f1 score of the model used is :red[{round(f1_score(y_cat_test, y_cat_pred, average='weighted'), 4)}]")

        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)


        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)


        # Streamlit app layout
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gender_vo,age_input,hypertension_vo,heart_disease_vo,ever_married_vo,work_type_vo,residence_type_vo,avg_glucose_level_input,bmi_input,smoking_status_vo]] ))
        if pred_res==0:
            pred_res="No"
        elif pred_res==1:
            pred_res="Yes"

        st.markdown(f"### Your predicted stroke status :blue[is:  {pred_res}] based on the input values.")






    if classifier_type=="Decision Tree":

        modell = DecisionTreeClassifier()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        st.markdown(f"## The score of the model used is :red[{round(sc,4)}]")



        cv_nums= st.number_input("Insert the value Cross Folds for validation", value=4, placeholder="Type a number..........", min_value=2,max_value=30, step=1)
        cv_sc = cross_val_score(modell, features, labels, cv=cv_nums)
        st.markdown("## The cross validation scores of the model used are ")
        for i in range(cv_nums):
            st.markdown(f"## :red[{round(cv_sc[i],4)}]")

        y_cat_pred = modell.predict(x_cat_test)
        st.markdown(f"## The f1 score of the model used is :red[{round(f1_score(y_cat_test, y_cat_pred, average='weighted'), 4)}]")

        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)


        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)


        # Streamlit app layout
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gender_vo,age_input,hypertension_vo,heart_disease_vo,ever_married_vo,work_type_vo,residence_type_vo,avg_glucose_level_input,bmi_input,smoking_status_vo]] ))
        if pred_res==0:
            pred_res="No"
        elif pred_res==1:
            pred_res="Yes"

        st.markdown(f"### Your predicted stroke status :blue[is:  {pred_res}] based on the input values.")

with tabs[8]:
    ## Classifiers Overview
    st.markdown("## :red[Classifiers Overview] \n"

                    "Comparing K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM)," 
                    "Random Forest, and Decision Trees reveals distinct characteristics and suitability for" 
                    "various scenarios:"

                    "\n - **K-Nearest Neighbors (KNN)**: This method shines with its simplicity and effectiveness"
                    "  in handling small datasets. KNN is intuitive – it classifies based on the closest data"
                    "  points. However, its performance drops with large, high-dimensional datasets, as it"
                    "  becomes computationally demanding."

                    "\n - **Logistic Regression**: Renowned for its straightforward approach in binary"
                    "  classification problems, logistic regression is highly interpretable. It excels when"
                    "  relationships in data are linear but might falter with complex, non-linear data structures."

                    "\n - **Support Vector Machines (SVM)**: SVM stands out in handling complex, high-dimensional"
                    " data. It's particularly adept at text and image classification tasks, though its"
                      "interpretability can be challenging, especially with non-linear kernels. SVM's"
                      "effectiveness depends heavily on the right parameter tuning."

                    "\n - **Random Forest**: As an ensemble method of decision trees, Random Forest is robust"
                      "against overfitting and versatile, performing well across a range of classification"
                      "and regression tasks. While it's more complex and less interpretable than a single"
                      "decision tree, it offers insights into feature importance."

                    "\n - **Decision Trees**: These are the bedrock of simplicity and interpretability. Decision"
                    " trees work well for both classification and regression but are prone to overfitting."
                    " They are often used in scenarios where understanding the model's decision path is crucial."

                    "\n In summary, each classifier has its niche: KNN for simplicity in small datasets,"
                    "Logistic Regression for binary outcomes with linear relations, SVM for high-dimensional"
                    "data, Random Forest for robust and versatile applications, and Decision Trees for clear,"
                    "interpretable results. The choice depends on the specific needs of the data and the"
                    "problem at hand.\n"

                    "### :red[Regression Overview] \n"

                    "Polynomial regression is used to model the relationship between an independent variable"
                    "`x` and a dependent variable `y` using a polynomial expression. The degree of the"
                    "polynomial influences the model's complexity and fit to the data.\n"

                    "#### Types of Polynomial Regression:\n"

                    "\n :blue[1. **Linear (1st Degree)**:]"
                    " \n - **Formula**: $ y = a_0 + a_1x $"
                    "\n- **Shape**: Straight line"
                    "\n- **Application**: Best for linear relationships in data\n"

                    "\n2. :blue[**Quadratic (2nd Degree)**:]"
                    "\n- **Formula**: $ y = a_0 + a_1x + a_2x^2 $"
                    " \n - **Shape**: U-shaped or inverted U curve"
                    "  \n - **Application**: Suitable for data with acceleration or deceleration patterns\n"

                    "\n3. :blue[**Cubic (3rd Degree)**:]\n"
                       "\n- **Formula**: $ y = a_0 + a_1x + a_2x^2 + a_3x^3 $"
                       "\n- **Shape**: S-shaped curve"
                       "\n- **Application**: Effective for complex data not well-modeled by linear or quadratic"
                        " curves\n"

                    "\n4. :blue[**Higher Degrees**:]\n"
                       "\n- **Formula**: Extends to $ a_nx^n $ for $ n > 3 $"
                       "\n- **Shape**: Complex, with multiple peaks and troughs"
                       "\n- **Application**: Can model intricate patterns but may overfit with high degrees\n"

                    "\n#### Considerations:"

                    "\n- Increasing the polynomial degree adds complexity, allowing the model to fit a broader"
                    " range of data patterns."
                    "\n- Higher degrees can lead to overfitting, capturing noise rather than the underlying trend."
                    "\n- Selecting the polynomial degree involves balancing complexity with the risk of overfitting"
                    " to best represent the data."

                    "\n## :red[Principal Component Analysis (PCA) Overview]"

                    "\nPrincipal Component Analysis (PCA) is a statistical method used for dimensionality reduction"
                    "in data analysis and machine learning. It simplifies complex, high-dimensional data while"
                    "preserving trends and patterns."

                    "\n### Key Aspects"
                    "\n- **Dimensionality Reduction**: Transforms high-dimensional data into fewer dimensions"
                      "(principal components) without losing significant information."
                    "\n- **Principal Components**: Orthogonal axes that maximize the variance in the data. The"
                      "first component captures the most variance, and each subsequent component captures less."
                    "\n- **Standardization**: Often involves standardizing the data so each feature contributes"
                " equally."
                    "\n- **Eigenvalues and Eigenvectors**: PCA involves computing these to determine the principal"
                      "components and their importance."
                    "\n- **Applications**: Used for data visualization, noise reduction, and feature extraction."
                    "\n- **Limitations**: Not effective for non-linear data and may result in loss of interpretable"
                    " features."

                    "\nPCA is particularly valuable in reducing data complexity, aiding in visual analysis, and"
                    "improving the efficiency of machine learning models."

                    "\n## :red[Stroke Dataset Analysis Summary]"

                    "\n### Key Correlations with Stroke"
                    "\n- **Age**: Exhibits the highest correlation (0.246), indicating a stronger prevalence of"
                      "stroke in older age groups."
                    "\n- **Heart Disease**: Shows a notable positive correlation, suggesting a significant risk factor."
                    "\n- **Average Glucose Level**: Positively correlated, highlighting its importance in stroke risk."
                    "\n- **Hypertension**: Another significant factor with a positive correlation."
                    "\n- **BMI**: Presents a weaker correlation but still relevant in assessing stroke risk."

                    "\n### Visual Analysis and Insights"
                    "\n1. **Gender vs Stroke**:"
                       "\n- Visualization of stroke cases in males and females. It helps assess if there's a"
                         "notable gender-based difference in stroke occurrence."

                    "\n2. **Age Distribution of Stroke Patients**:"
                       "\n- The histogram of ages in stroke patients underlines the increased occurrence in"
                         "certain age groups, particularly older individuals."

                    "\n3. **Hypertension vs Stroke**:"
                       "\n- Comparison of stroke cases in individuals with and without hypertension, indicating"
                         "the impact of hypertension on stroke risk."

                    "\n4. **Heart Disease vs Stroke**:"
                       "\n- Similar to hypertension, this shows how the presence of heart disease relates to"
                         "stroke occurrence."

                    "\n5. **Smoking Status vs Stroke**:"
                       "\n- This comparison across different smoking statuses (never smoked, formerly smoked, smokes)"
                         "provides insights into smoking as a potential risk factor."

                    "\n6. **Residence Type vs Stroke**:"
                       "\n- The comparison between urban and rural residents offers perspective on whether living"
                         "environment influences stroke risk."

                    "\n### Conclusions"
                    "\n- **Age** is a primary factor in stroke risk, with older individuals showing higher prevalence."
                    "\n- Medical conditions like **heart disease** and **hypertension** significantly increase"
                      "stroke risk."
                    "\n- Lifestyle factors such as **smoking** and perhaps even **work type** play roles in stroke"
                      "occurrence."
                    "\n- The **average glucose level** is an important metric to consider in stroke risk assessment."
                    "\n- The role of **BMI** is less pronounced but should not be overlooked."
                    "\n- There's no clear evidence from this dataset suggesting a significant impact of **gender**"
                " or **residence type** on stroke risk."

                    "\nThese findings collectively enhance the understanding of stroke risk factors, indicating that"
                    "a combination of age, medical history, and lifestyle choices contributes to the overall risk"
                    "profile for stroke.")
