import streamlit as st

def execute_academy_step(step, manual_content):
    """
    Handles the 3-Step Energy Academy Process.
    Separated from the main chatbot to prevent crashes.
    """
    if not manual_content:
        st.warning("⚠️ Please upload a technical manual (.md) to begin the Academy steps.")
        return

    # --- FIXED: Added the missing except block for the main try ---
    try:
        if step == "Step 1: Fixed Presentation":
            st.markdown("### 📊 Maps Academy: Professional Presentation")
            st.write("---")
            
            if "current_presentation" not in st.session_state:
                st.info("Sree is forging your 3-Level Presentation... please wait.")
            
            if "current_presentation" in st.session_state:
                st.markdown(st.session_state.current_presentation)
                st.success("📝 **Next Step:** Move to 'Step 2: The Tutor'.")
            
        elif step == "Step 2: The Tutor":
            st.markdown("### 👨‍🏫 Maps Academy: Expert Tutor Mode")
            st.success("Sree is now focused strictly on your technical manual.")
            
            with st.expander("💡 How to talk to your Tutor"):
                st.write("""
                * *Novice:* 'Can you explain the basic parts?'
                * *Intermediate:* 'What are the main risks?'
                * *Professional:* 'How can I optimize for wealth?'
                """)
            st.info("⚠️ Sree will only use wealth found in your uploaded manual.")
            
        elif step == "Step 3: Graduation Quiz":
            st.markdown("### 🎓 Maps Academy: Skill Validation")
            
            if "current_quiz" not in st.session_state:
                st.info("Sree is preparing your validation questions...")
                st.write("Please click 'Generate Quiz' to begin.")
            
            if "current_quiz" in st.session_state:
                st.markdown(st.session_state.current_quiz)
                
                if st.button("🗑️ Clear Quiz and Return to Chat", key="module_close_btn"):
                    del st.session_state.current_quiz
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Academy Logic Error: {str(e)}")
