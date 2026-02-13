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
            st.markdown("### 📊 Energy Academy: Professional Presentation")
            # This triggers a specific structured summary
            st.write("---")
            # Logic for presentation goes here
            
        elif step == "Step 2: The Tutor":
            st.markdown("### 👨‍🏫 Sree: Expert Energy Tutor")
            st.info("I am now focused strictly on your uploaded manual. Ask me anything!")
            
        elif step == "Step 3: Graduation Quiz":
            st.markdown("### 🎓 Maps Academy: Skill Validation")
            
            if "current_quiz" not in st.session_state:
                # This is the 'Genie' move: 
                # We ask the AI to generate questions BASED on the uploaded manual
                with st.spinner("Sree is generating your custom validation quiz..."):
                    # This calls your AI function (replace 'get_ai_response' with your actual function name)
                    quiz_content = get_ai_response(f"Generate 3 difficult multiple-choice questions based on this manual: {manual_content}")
                    st.session_state.current_quiz = quiz_content
            
            st.markdown(st.session_state.current_quiz)
            
            if st.button("🗑️ Clear Quiz and Return to Chat", key="module_close_btn"):
                del st.session_state.current_quiz
                st.rerun()
            else:
                st.write("Click 'Generate Graduation Quiz' in the sidebar to start.")

    except Exception as e:
        st.error(f"Academy Module Error: {e}")
