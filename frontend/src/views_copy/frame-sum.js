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
import axiosBase from 'axiosConfig';
import link from 'assets/images/paperclip-solid.svg'
import fileicon from 'assets/external/fileicons.png'

const XLSX = require('xlsx');


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
  const nonOpenCodingAnalysisOptions = ['ALL', 'summary', 'Sentiment analysis', 'Keyword Extraction'];


  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [inputPlaceholder, setInputPlaceholder] = useState('Generate message..'); // 기본 플레이스홀더 설정


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

  const sendMessage = async () => {
    if (userInput.trim() === '') return;

    const updatedChatHistory = [...chatHistory, { text: userInput, user: true }];
    
    setChatHistory(updatedChatHistory);
    setUserInput('');

    let reply = userInput;
    if (openCoding) {
      reply = await axiosBase.post('oc/text', {
        text: userInput,
        pos: analysisType === '긍정 질문' ? true : false,
        model: selectedModel, // 사용자가 선택한 프롬프트 값
      }).then((res) => { return res.data });
    } else {
      // 분석 방법 선택일 경우
      reply = await axiosBase.post('sum/text', {
        text: userInput,
        task: analysisType,
        model: selectedModel
      }).then((res) => { return res.data });
    }


    setTimeout(() => {
      const updatedChatHistoryWithReply = [...updatedChatHistory, { text: reply, user: false }];
      setChatHistory(updatedChatHistoryWithReply);
    }, 500);
  };


  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    if (file.type === 'text/csv' || file.type === 'application/vnd.ms-excel' || file.name.endsWith('.xlsx')) {

      const reader = new FileReader();

      const updatedChatHistory = [...chatHistory, { text: <><img alt="file-icon" src={fileicon} style={{ width: "20px", height: "20px", marginRight: "15px" }} />{file.name}</>, user: true }];
    
      setChatHistory(updatedChatHistory);

      const formData = new FormData();
      formData.append("file", file);

      if (openCoding) {
      let reply = await axiosBase.post('oc/file', formData,
        {
          params: {model: selectedModel , pos: analysisType === '긍정 질문' ? "True" : "False"},
          headers: {"Content-Type": "multipart/form-data"},
          responseType: 'blob'
        }
      ).then((res) => { 
        const blob = new Blob([res.data], { type: res.headers['content-type'] }); 
        const fileObjectUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = fileObjectUrl;
        link.setAttribute('download', file.name);
        link.click();
        return '파일 생성이 완료되었습니다.'
      })
    }else{
      let reply = await axiosBase.post('sum/file', formData,
        {
          params: {model: selectedModel , task: analysisType},
          headers: {"Content-Type": "multipart/form-data"},
          responseType: 'blob'
        }
      ).then((res) => { 
        const blob = new Blob([res.data], { type: res.headers['content-type'] }); 
        const fileObjectUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = fileObjectUrl;
        link.setAttribute('download', file.name);
        link.click();
        return '파일 생성이 완료되었습니다.'
      })
    }
        

      setTimeout(() => {
        const updatedChatHistoryWithReply = [...updatedChatHistory, { text: reply, user: false }];
        setChatHistory(updatedChatHistoryWithReply);
      }, 500);

      /*
      reader.onload = (event) => {
        const fileContent = event.target.result;
        // xlsx 파일 처리를 위해 xlsx 라이브러리 사용
        const workbook = XLSX.read(fileContent, { type: 'binary' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
      };
      reader.readAsBinaryString(file);
      */
    } else {
      alert('올바른 파일 형식이 아닙니다. xlsx 파일이나 csv 파일을 선택해주세요.');
    }
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
            <div key={index} className={`frame-sum-${item.user ? 'user' : 'model'}-message`}>
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
                {!item.user && <span className="frame-sum-text11"></span>}
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

                <label htmlFor="file-input">
                  &nbsp;&nbsp;
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" style={{ display: "inline-block", width: "30px", height: "30px", cursor: "pointer" ,  paddingLeft: "0px", fill: "#4d4d4d" }}>
                    <path d="M364.2 83.8c-24.4-24.4-64-24.4-88.4 0l-184 184c-42.1 42.1-42.1 110.3 0 152.4s110.3 42.1 152.4 0l152-152c10.9-10.9 28.7-10.9 39.6 0s10.9 28.7 0 39.6l-152 152c-64 64-167.6 64-231.6 0s-64-167.6 0-231.6l184-184c46.3-46.3 121.3-46.3 167.6 0s46.3 121.3 0 167.6l-176 176c-28.6 28.6-75 28.6-103.6 0s-28.6-75 0-103.6l144-144c10.9-10.9 28.7-10.9 39.6 0s10.9 28.7 0 39.6l-144 144c-6.7 6.7-6.7 17.7 0 24.4s17.7 6.7 24.4 0l176-176c24.4-24.4 24.4-64 0-88.4z"/>
                  </svg>
                </label>
                <input type="file" id="file-input" style={{ display: "none" }} />


                  
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
