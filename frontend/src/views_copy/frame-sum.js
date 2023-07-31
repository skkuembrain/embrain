import React, { useState, useRef, useEffect } from 'react';
import logo from "assets/external/embrainlogo-200h-200h.png";
import 'views_copy/frame-sum-copy.css'


import link from 'assets/images/paperclip-solid.svg'


const FrameSum = () => {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [inputPlaceholder, setInputPlaceholder] = useState('Generate message..'); // 기본 플레이스홀더 설정

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (event) => {
      const fileContent = event.target.result;
      // 여기서 fileContent를 처리하거나 전송할 수 있습니다.
      setUserInput(fileContent); // 입력창에 파일 내용을 설정합니다.
    };

    if (file.type === 'text/csv' || file.type === 'application/vnd.ms-excel') {
      reader.readAsText(file);
    } else {
      alert('올바른 파일 형식이 아닙니다. xlsx 파일이나 csv 파일을 선택해주세요.');
    }
  };

  const chatHistoryRef = useRef(null);
  const inputRef = useRef(null);

  const getCurrentTimeInKorean = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1;
    const date = now.getDate();
    const hour = now.getHours();
    const minute = now.getMinutes();

    return `${year}년 ${month}월 ${date}일 ${hour}시 ${minute}분`;
  };

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const sendMessage = () => {
    if (userInput.trim() === '') return;

    const updatedChatHistory = [...chatHistory, { text: userInput, user: true }];
    setChatHistory(updatedChatHistory);
    setUserInput('');

    const reply = '모델이 생성한 답변입니다.';
    setTimeout(() => {
      const updatedChatHistoryWithReply = [...updatedChatHistory, { text: reply, user: false }];
      setChatHistory(updatedChatHistoryWithReply);
    }, 500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  const handleInputFocus = () => {
    setInputPlaceholder(''); // 입력창에 포커스되면 기본 플레이스홀더를 제거하여 빈 상태로 설정
  };

  useEffect(() => {
    const chatHistoryDiv = chatHistoryRef.current;
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
  }, [chatHistory]);

  return (
    <div className="frame-sum-frame">
      <div className="frame-sum-dialog" ref={chatHistoryRef}>
        {chatHistory.map((item, index) => (
          <div key={index} className={`frame-sum-${item.user ? 'user' : 'model'} ${item.user ? 'user-message' : 'model-message'}`}>
            <div className={`frame-sum-${item.user ? 'date-user-name' : 'embrain-model'}`}>
              <div className={`frame-sum-${item.user ? 'text-date' : 'logo-embrain'}`}>
                {item.user ? getCurrentTimeInKorean() : <img alt="logo" src={logo} className="frame-sum-logo-embrain1" style={{ width: "20px", height: "20px" }} />}
              </div>
              {!item.user && <span className="frame-sum-text-user-name">Model</span>}
            </div>
            <div className={`frame-sum-${item.user ? 'users-input-data' : 'model-output-data'}`}>
              <span className={`frame-sum-${item.user ? 'what-user-text' : 'text10'}`}>
                {item.text}
              </span>
              {!item.user && <span className="frame-sum-text11"><span>2 mb</span></span>}
            </div>
          </div>
        ))}
      </div>
      <div className="frame-sum-user-input">
        <div className="frame-sum-input">
          <div className="frame-sum-input-box">
            <div className="frame-sum-inner-input-box">
              
              <input
                  type="file"
                  id="file-input"
                  accept=".csv, .xlsx"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
                <img
                  alt="unnamed932w7rR4ftransformed11226"
                  src={link}
                  className="frame-sum-logo-embrain1"
                />
                {/* <label htmlFor="file-input" className="frame-sum-file-button">
                  파일 선택
                </label> */}
                
              <textarea
                placeholder={inputPlaceholder} // 상태 변수에 저장된 플레이스홀더를 적용
                autoFocus
                className="frame-sum-textarea textarea"
                value={userInput}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                onFocus={handleInputFocus} // 입력창이 포커스되면 기본 플레이스홀더를 제거
                ref={inputRef}
                style={{ height: "5em", resize: "none" }}
              ></textarea>
              <button className="frame-sum-send-button" onClick={sendMessage}>
                <span>전송</span>
              </button>
            </div>
          </div>
          <div className="frame-sum-cancel-guide">
            <span className="frame-sum-text-cancel-guide">
              <span>ESC or Click to cancel</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FrameSum;
