import streamlit as st

def execute_academy_step(step, manual_content):
    """
    Handles the 3-Step Energy Academy Process.
    Separated from the main chatbot to prevent crashes.
    """
    if not manual_content:
        st.warning("⚠️ Please upload a technical manual (.md) to begin the Academy steps.")
        return

    try:
        if step == "Step 1: Fixed Presentation":
            st.markdown("### 📊 Maps Academy: Professional Presentation")
            st.write("---")
            
            # We check if Sree has already 'Forged' the presentation for this file
            if "current_presentation" not in st.session_state:
                st.info("Sree is forging your 3-Level Presentation... please wait.")
                # When you click the 'Speak with Sree' button, use this prompt:
                # "Based on the uploaded manual, create a Fixed Presentation with 3 levels: 
                # Level 1 (Novice), Level 2 (Intermediate), and Professional (Wealth-Saving).
                # Use bold headers and focus on energy efficiency."
            
            if "current_presentation" in st.session_state:
                st.markdown(st.session_state.current_presentation)
                
                # A reminder for the student
                st.success("📝 **Next Step:** Move to 'Step 2: The Tutor' to ask specific questions about this material.")
            
        elif step == "Step 2: The Tutor":
            st.markdown("### 👨‍🏫 Sree: Expert Energy Tutor")
            st.info("I am now focused strictly on your uploaded manual. Ask me anything!")
            
        elif step == "Step 3: Graduation Quiz":
            st.markdown("### 🎓 Maps Academy: Skill Validation")
            
            # We check if a quiz already exists for this session
            if "current_quiz" not in st.session_state:
                st.info("Sree is preparing your validation questions based on the manual...")
                # This is a placeholder - the actual generation happens in your main chatbot
                st.write("Please click 'Generate Quiz' to begin.")
            
            if "current_quiz" in st.session_state:
                st.markdown(st.session_state.current_quiz)
                
                if st.button("🗑️ Clear Quiz and Return to Chat", key="module_close_btn"):
                    del st.session_state.current_quiz
                    st.rerun()
            else:
                st.write("Click 'Generate Graduation Quiz' in the sidebar to start.")

    except Exception as e:
        st.error(f"Academy Module Error: {e}")
