import streamlit as st

def execute_academy_step(step, manual_content):
    """
    Handles the 3-Step Process: Presentation, Tutor, Quiz.
    This file is separate to keep main.py clean and safe.
    """
    if not manual_content:
        st.info("💡 **Welcome to the Energy Academy.** Please upload a manual in the sidebar to begin your training.")
        return

    # Logic for Step 3: Graduation Quiz
    if step == "Step 3: Graduation Quiz":
        st.subheader("🎓 Graduation Quiz & Skill Validation")
        
        if "current_quiz" in st.session_state:
            st.markdown(st.session_state.current_quiz)
            st.markdown("---")
            # This is the "Sticky" button we fought for! 
            # It lives here now, away from the main chatbot code.
            if st.button("🗑️ Clear Quiz and Return to Chat", key="academy_close_btn"):
                del st.session_state.current_quiz
                st.rerun()
        else:
            st.warning("Please click 'Generate Graduation Quiz' in the sidebar to start.")

    # Logic for Step 1: Presentation
    elif step == "Step 1: Fixed Presentation":
        st.subheader("📊 Professional Module Summary")
        st.write("---")
        # Sree will use the manual_content to fill this in
        st.info("Study the summary above before moving to the Tutor or Quiz.")

    # Logic for Step 2: The Tutor
    elif step == "Step 2: The Tutor":
        st.subheader("👨‍🏫 Sree's Expert Tutoring")
        st.success("Tutor mode active. Ask me anything about the technical manual!")
