import React, { useState, useRef, useEffect } from 'react';
import logo from "assets/external/embrainlogo-200h-200h.png";
import 'views_copy/frame-sum-copy.css'
import {
  FormControl,
  InputAdornment,
  OutlinedInput,
  Avatar,
  AvatarGroup,
  Box,
  Button,
  Grid,
  List,
  ListItemAvatar,
  ListItemButton,
  ListItemSecondaryAction,
  ListItemText,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

import axios from 'axios';


import link from 'assets/images/paperclip-solid.svg'
import { is } from '../../../../../../AppData/Local/Microsoft/TypeScript/5.1/node_modules/@babel/types/lib/index';


const FrameSum = () => {


  const [selectedModel, setSelectedModel] = useState('kogpt2');
  const [openCoding, setOpenCoding] = useState(true);
  const [analysisType, setAnalysisType] = useState(openCoding ? '긍정 질문' : 'ALL');

  const [filePreview, setFilePreview] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [answer, setAnswer] = useState(''); // 답변을 저장할 상태 변수

  const models = [
    { value: 'kogpt2', label: 'KoGPT2' },
    { value: 'trinity', label: 'Trinity' },
    { value: 'polyglot', label: 'Polyglot' },
  ];

  const openCodingOptions = [
    { value: true, label: '정량분석' },
    { value: false, label: '정성분석' },
  ];

  const openCodingAnalysisOptions = ['긍정 질문', '부정 질문'];
  const nonOpenCodingAnalysisOptions = ['ALL', 'Summary', 'Sentiment analysis', 'Keyword Extraction'];


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



  useEffect(() => {
    const chatHistoryDiv = chatHistoryRef.current;
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
  }, [chatHistory]);
  
  useEffect(() => {
    if (!openCoding) {
      setAnalysisType('ALL');
    }
    else{
      setAnalysisType('긍정 질문');
    }
  }, [openCoding]);
  

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
    

    // const reply = '모델이 생성한 답변입니다.';
    // const reply = await axios.post('localhost:8000/oc/text', {
    //   text: '맵고 달고 짜요',
    //   pos: 'False',
    //   model: 'trinity'
    // })
  


    if (openCoding) {
      const reply = axios.post('http://localhost:8000/api/data', {
        text: {userInput},
        pos: analysisType === '긍정 질문' ? true : false,
        model: {selectedModel}, // 사용자가 선택한 프롬프트 값
      });
    } else {
      // 분석 방법 선택일 경우
      const reply = axios.post('http://localhost:8000/api/data', {
        text: {userInput},
        task: {analysisType},
        model: {selectedModel}
      });
    }

    setChatHistory(updatedChatHistory);
    setUserInput('');
    

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

    <Grid container rowSpacing={4.5} columnSpacing={2.75} alignItems="center">

    <Grid item xs={12} md={1} lg={12}>
      <Grid container alignItems="center" justifyContent="space-between">
        <FormControl variant="outlined" size="small" style={{width: "33%"}}>
          <TextField
            select
            label="모델 선택"

            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </TextField>
        </FormControl>

        <FormControl variant="outlined" size="small" style={{width: "33%"}}>
          <TextField
            select
            label="분석 선택"
            value={openCoding}
            onChange={(e) => setOpenCoding(e.target.value)}
          >
            {openCodingOptions.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </TextField>
        </FormControl>

        {openCoding !== null && (
          <FormControl variant="outlined" size="small" style={{width: "33%"}}>
            <TextField
              select
              label={openCoding ? 'Opencoding 프롬프트 선택' : '분석 방법 선택'}
              value={analysisType}
              onChange={(e) => setAnalysisType(e.target.value)}
            >
              {openCoding
                ? openCodingAnalysisOptions.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))
                : nonOpenCodingAnalysisOptions.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
            </TextField>
          </FormControl>
        )}

    <div className="frame-sum-frame">
      <div className='frame-dialog' ref={chatHistoryRef}>
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
                  <label htmlFor="file-input" className="frame-sum-file-button">
                    파일 선택
                  </label>
                  
                <textarea
                  placeholder={inputPlaceholder} // 상태 변수에 저장된 플레이스홀더를 적용
                  autoFocus
                  className="frame-sum-textarea textarea"
                  value={userInput}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  onFocus={handleInputFocus} // 입력창이 포커스되면 기본 플레이스홀더를 제거
                  ref={inputRef}
                  style={{ height: "5em", resize: "none",  border: "1px solid #ddd"  }}
                ></textarea>
                <button className="frame-sum-send-button" onClick={sendMessage}>
                  <span>전송</span>
                </button>
              </div>
            </div>
            <div className="frame-sum-cancel-guide">
            </div>
          </div>
        </div>
      </div>
    
        
      </Grid>
    </Grid>
  </Grid>


  
  );
};

export default FrameSum;
