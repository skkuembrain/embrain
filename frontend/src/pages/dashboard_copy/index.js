import { useState } from 'react';
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

const avatarSX = {
  width: 36,
  height: 36,
  fontSize: '1rem',
};

const actionSX = {
  mt: 0.75,
  ml: 1,
  top: 'auto',
  right: 'auto',
  alignSelf: 'flex-start',
  transform: 'none',
};

import Frame159 from 'views_copy/frame-sum'


const DashboardDefaultCopy = () => {
  const [selectedModel, setSelectedModel] = useState('kogpt2');
  const [openCoding, setOpenCoding] = useState(true);
  const [analysisType, setAnalysisType] = useState('긍정 질문');
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

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target.result);
          setFilePreview(jsonData);
          setUploadedFileName(file.name);
        } catch (error) {
          setFilePreview(null);
          setUploadedFileName('');
        }
      };
      reader.readAsText(file);
    } else {
      setFilePreview(null);
      setUploadedFileName('');
    }
  };

  const handleSend = () => {
    // 이 부분에 답변 생성 로직을 추가합니다.
    // 여기에서 llm 모델을 사용하여 답변을 생성하고, setAnswer로 상태 변수에 저장합니다.
    const generatedAnswer = '여기에 답변을 생성하는 로직을 추가하세요.'; // 임시로 답변을 생성하는 예시입니다.
    setAnswer(generatedAnswer);
  };

  return (
    <Grid container rowSpacing={4.5} columnSpacing={2.75} alignItems="center">

      <Grid item xs={12} md={1} lg={12}>
        <Grid container alignItems="center" justifyContent="space-between">
          <FormControl fullWidth variant="outlined" size="small">
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
          <Box sx={{ height: 50 }} />

          <FormControl fullWidth variant="outlined" size="small">
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
          <Box sx={{ height: 100 }} />

          {openCoding !== null && (
            <FormControl fullWidth variant="outlined" size="small">
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


        <Grid item xs={12} md={1} lg={12}>
            <Grid container alignItems="center" justifyContent="space-between">
              <Frame159/>
            </Grid>
          </Grid>


  

          
        </Grid>
      </Grid>
    </Grid>
  );
};

export default DashboardDefaultCopy;
