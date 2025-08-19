import streamlit as st
import joblib
import pandas as pd
import ast


# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title=" Matchmaking AI", layout="centered")
st.title(" Matchmaking Compatibility Predictor")

st.markdown("Fill in the details for **two people** to check how compatible they are!")

# ---------- Shared Lists ----------
genders = ["Male", "Female", "Other"]
looking_for = ["Casual Dating", "Friendship", "Marriage", "Long-term Relationship"]
children_opts = ["Yes", "No", "Maybe"]
edu_levels = ["High School", "Bachelor's Degree", "Master's Degree", "Ph.D."]
occupations = ["Engineer", "Doctor", "Artist", "Teacher", "Entrepreneur", "Business Owner", "Student", "Social Media Influencer"]
usage_freq = ["Daily", "Weekly", "Monthly"]
interest_options = ["Music", "Movies", "Reading", "Sports", "Travel", "Cooking", "Hiking"]

def user_form(label):
    st.subheader(label)
    age = st.slider(f"{label} - Age", 18, 60, 25)
    gender = st.selectbox(f"{label} - Gender", genders)
    height = st.slider(f"{label} - Height (feet)", 4.5, 7.0, 5.5)
    interest = st.multiselect(f"{label} - Interests", interest_options)
    goal = st.selectbox(f"{label} - Looking For", looking_for)
    children = st.selectbox(f"{label} - Children", children_opts)
    edu = st.selectbox(f"{label} - Education Level", edu_levels)
    occ = st.selectbox(f"{label} - Occupation", occupations)
    usage = st.selectbox(f"{label} - App Usage", usage_freq)
    return {
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Interests": str(interest),  # stored as string
        "Looking For": goal,
        "Children": children,
        "Education Level": edu,
        "Occupation": occ,
        "Frequency of Usage": usage
    }

# ---------- User Inputs Side-by-Side with Spacer ----------
col1, spacer, col2 = st.columns([1, 0.2, 1])  # Side-by-side layout with gap

with col1:
    user1 = user_form("👤 User A")

with col2:
    user2 = user_form("👤 User B")

# ---------- Jaccard Similarity ----------
def jaccard_similarity(list1, list2):
    set1, set2 = set(ast.literal_eval(list1)), set(ast.literal_eval(list2))
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

# ---------- Predict Button and Result ----------
st.markdown("---")


# Centered prediction section
center = st.container()
with center:
    if st.button("Predict Compatibility", use_container_width=True):
        input_row = pd.DataFrame([{
            "User1_Gender": user1["Gender"],
            "User2_Gender": user2["Gender"],
            "User1_Age": user1["Age"],
            "User2_Age": user2["Age"],
            "User1_LookingFor": user1["Looking For"],
            "User2_LookingFor": user2["Looking For"],
            "User1_Children": user1["Children"],
            "User2_Children": user2["Children"],
            "User1_Education": user1["Education Level"],
            "User2_Education": user2["Education Level"],
            "User1_Occupation": user1["Occupation"],
            "User2_Occupation": user2["Occupation"],
            "User1_Usage": user1["Frequency of Usage"],
            "User2_Usage": user2["Frequency of Usage"],
            "User1_Height": user1["Height"],
            "User2_Height": user2["Height"],
            "Interest_Similarity": jaccard_similarity(user1["Interests"], user2["Interests"])
        }])

        score = model.predict(input_row)[0]
        st.success(f"Predicted Compatibility Score: **{score:.2f}%**")

        # Interpretation
        if score > 80:
            st.balloons()
            st.markdown("### 💍 Perfect Match!")
        elif score > 60:
            st.markdown("### 💕 Strong Compatibility")
        elif score > 40:
            st.markdown("### 🙂 Might Work with Effort")
        else:
            st.markdown("### ⚠️ Not Very Compatible")
