import streamlit as st
import pandas as pd
import time
import uuid
from google import genai
from google.genai.errors import ResourceExhaustedError, APIError

# --- 상수 및 설정 ---
APP_TITLE = "🛍️ 구매 결정 도우미 챗봇"
# 허용되는 모델 목록 (gemini-2.0-flash 기본 선택)
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash", 
    "gemini-2.5-pro",
    "gemini-2.0-pro"
]
MAX_CONTEXT_TURNS = 6 # 유지할 최대 턴 수 (유저 + 모델 = 2개의 메시지, 즉 12개 파트)
MAX_HISTORY_PARTS = MAX_CONTEXT_TURNS * 2 
MAX_RETRIES = 3 # 429 에러 발생 시 최대 재시도 횟수

# --- 시스템 프롬프트 정의 (친절한 응대 및 정보 수집 유도) ---
SYSTEM_PROMPT = """
당신의 역할은 쇼핑몰 구매 과정에서 고민하는 고객을 돕는 구매 결정 도우미 챗봇입니다.
친절하고 공감 어린 말투로 고객의 상황을 경청하고 대화해주세요. 고객이 편안하게 고민을 털어놓을 수 있도록 정중하게 대응해야 합니다.

## 고객 응대 필수 목표:
1.  **공감 및 정보 수집:** 사용자가 고민하는 사항(무엇을/언제/어디서/어떻게)을 구체적으로 정리하여 수집하고, 이 정보를 고객 응대 담당자에게 전달할 것임을 안내하세요. 질문을 통해 구체적인 정보를 얻으려 노력해야 합니다.
2.  **연락처 요청:** 마지막 응답 시, 담당자 확인 후 회신을 위해 반드시 고객의 이메일 주소를 요청하세요. 이메일 주소 요청은 답변 제공의 필수 조건임을 명확히 안내해야 합니다.
3.  **연락처 거부 처리:** 만일 고객이 연락처 제공을 원치 않거나 거부하는 경우, "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요."라고 정중히 안내하며 대화를 종료하세요. 이 안내 이후에는 추가 답변을 제공하지 마세요.
"""

# --- 세션 상태 초기화 ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    # 전체 대화 로그를 저장 (API 호출에 사용되는 컨텍스트는 별도 관리)
    st.session_state.chat_history = []
if 'log_count' not in st.session_state:
    st.session_state.log_count = 0

# --- API 키 설정 및 클라이언트 생성 ---
def get_gemini_client():
    """Streamlit Secrets 또는 사용자 입력에서 API 키를 가져오고 클라이언트를 반환합니다."""
    # 1. st.secrets에서 키를 시도
    api_key = st.secrets.get('GEMINI_API_KEY')

    if not api_key:
        # 2. st.secrets에 키가 없으면, 사이드바에서 사용자 입력을 받음
        with st.sidebar:
            st.warning("`st.secrets['GEMINI_API_KEY']`가 설정되지 않았습니다. 임시로 API 키를 입력해주세요.", icon="⚠️")
            user_input_key = st.text_input("Gemini API Key", type="password", key="user_key_input")
            if user_input_key:
                api_key = user_input_key
            else:
                st.info("API 키를 입력해야 챗봇을 사용할 수 있습니다.")
                return None
    
    # 3. 클라이언트 생성 및 반환
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"API 클라이언트 초기화 중 오류 발생: {e}")
        return None

# --- UI 요소: 사이드바 설정 ---
def setup_sidebar():
    """사이드바에 모델 선택, 로그 설정, 세션 정보를 설정합니다."""
    st.sidebar.title("설정 및 로그")
    
    # 모델 선택
    selected_model = st.sidebar.selectbox(
        "사용할 Gemini 모델 선택:",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index("gemini-2.0-flash"),
        key="selected_model"
    )

    st.sidebar.markdown("---")

    # CSV 자동 기록 옵션
    st.sidebar.checkbox(
        "CSV에 대화 자동 기록 (선택)", 
        value=True, 
        key="auto_record_csv",
        help="새로운 턴이 발생할 때마다 전체 대화 로그를 CSV에 기록합니다."
    )
    
    # 세션 정보
    st.sidebar.markdown("### 세션 정보")
    st.sidebar.code(f"세션 ID: {st.session_state.session_id}")
    st.sidebar.info(f"선택된 모델: **{selected_model}**")

    st.sidebar.markdown("---")

    # 대화 초기화 버튼
    if st.sidebar.button("대화 초기화", type="primary"):
        st.session_state.chat_history = []
        st.session_state.log_count = 0
        st.session_state.session_id = str(uuid.uuid4())
        st.experimental_rerun()

    # 로그 다운로드 버튼
    if st.session_state.chat_history:
        log_data = [{"role": m["role"], "content": m["parts"][0]["text"]} 
                    for m in st.session_state.chat_history]
        df = pd.DataFrame(log_data)
        
        st.sidebar.download_button(
            label="전체 로그 다운로드 (CSV)",
            data=df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"chat_log_{st.session_state.session_id}.csv",
            mime="text/csv"
        )

# --- 메인 함수 ---
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption("고객님의 고민을 친절하게 듣고, 담당자에게 전달하여 맞춤형 회신을 준비해 드립니다. 이메일 주소를 요청드릴 수 있습니다.")
    
    setup_sidebar()
    client = get_gemini_client()

    if not client:
        # API 키가 없어 클라이언트 초기화 실패 시, 여기서 종료
        return

    # 대화 기록 표시
    for message in st.session_state.chat_history:
        # 'role'이 'user'면 'user', 'model'이면 'assistant'로 매핑하여 Streamlit 메시지 출력
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["parts"][0]["text"])

    # 사용자 입력 처리
    if user_prompt := st.chat_input("구매 결정에 대한 고민을 말씀해주세요."):
        
        # 1. 사용자 메시지를 기록 및 표시
        user_message_part = {"role": "user", "parts": [{"text": user_prompt}]}
        st.session_state.chat_history.append(user_message_part)
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. API 호출을 위한 컨텍스트 준비 (최근 6턴 유지)
        # st.session_state.chat_history의 마지막 12개 파트(6턴)만 컨텍스트로 사용
        context_for_api = st.session_state.chat_history[-MAX_HISTORY_PARTS:]
        
        # 3. API 호출
        with st.chat_message("assistant"):
            with st.spinner("담당자 검토를 위해 답변을 준비 중입니다..."):
                response_text = None
                for attempt in range(MAX_RETRIES):
                    try:
                        # contents 리스트에 시스템 프롬프트가 포함되지 않으므로, 
                        # generation_config를 사용하여 시스템 지침 전달
                        response = client.models.generate_content(
                            model=st.session_state.selected_model,
                            contents=context_for_api, # 컨텍스트 + 새 메시지
                            config=genai.types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT
                            )
                        )
                        response_text = response.text
                        break # 성공하면 루프 탈출
                        
                    except ResourceExhaustedError:
                        # 429 에러 처리 (재시도)
                        if attempt < MAX_RETRIES - 1:
                            wait_time = (2 ** attempt) + random.uniform(0, 1) # 지수 백오프
                            st.warning(f"트래픽 초과 (429) 발생! {wait_time:.2f}초 후 재시도합니다. (시도 {attempt + 1}/{MAX_RETRIES})")
                            time.sleep(wait_time)
                        else:
                            st.error("트래픽이 너무 많아 대화를 계속할 수 없습니다. 잠시 후 다시 시도하거나 대화를 초기화해주세요.")
                            response_text = "죄송합니다. 현재 서비스에 접속자가 많아 응답을 드릴 수 없습니다. 잠시 후 '대화 초기화' 버튼을 눌러 다시 시작해 주세요."
                            break # 재시도 횟수 초과

                    except APIError as e:
                        st.error(f"API 통신 오류 발생: {e}")
                        response_text = "API 통신 중 예기치 않은 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                        break
                    
                    except Exception as e:
                        st.error(f"알 수 없는 오류 발생: {e}")
                        response_text = "처리 중 알 수 없는 오류가 발생했습니다."
                        break

                # 4. 모델 응답 표시 및 기록
                if response_text:
                    st.markdown(response_text)
                    model_message_part = {"role": "model", "parts": [{"text": response_text}]}
                    st.session_state.chat_history.append(model_message_part)
                    st.session_state.log_count += 1
                    
                    # 5. CSV 자동 기록 (옵션)
                    if st.session_state.auto_record_csv:
                        # 전체 대화 목록을 사용하여 CSV 기록
                        log_data = [{"role": m["role"], "content": m["parts"][0]["text"]} 
                                    for m in st.session_state.chat_history]
                        df = pd.DataFrame(log_data)
                        df.to_csv(f"chat_log_{st.session_state.session_id}_auto.csv", index=False, encoding='utf-8-sig')

        # 다음 턴을 위해 페이지 새로고침
        st.experimental_rerun()

if __name__ == "__main__":
    main()