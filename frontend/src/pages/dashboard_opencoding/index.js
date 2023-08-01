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
import XLSX from 'xlsx'
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

const styles = {
  input: {
    width: '100%', // 너비를 원하는 값으로 설정하세요.
    height: '150px', // 원하는 고정 높이로 설정하세요.
    overflowY: 'auto', // 넘어갈 경우 스크롤되도록 설정합니다.
  },
};

const DashboardDefaultCopy = () => {
  const [selectedModel, setSelectedModel] = useState('kogpt2');
  const [openCoding, setOpenCoding] = useState(true);
  const [analysisType, setAnalysisType] = useState('긍정 질문');
  const [filePreview, setFilePreview] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');

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
          const xlsxData = XLSX.parse(e.target.result);
          // const data = new Uint8Array(e.target.result); // Convert the file data to Uint8Array
          // const workbook = XLSX.read(data, { type: 'array' }); // Read the workbook using xlsx library
          // const firstSheetName = workbook.SheetNames[0]; // Get the first sheet name
          // const worksheet = workbook.Sheets[firstSheetName]; // Get the worksheet
          // const xlsxData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }); // Convert worksheet to JSON
          setFilePreview(xlsxData);
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
    // 전송 로직을 추가하세요.
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

          <input
            accept=".xlsx"
            style={{ display: 'none' }}
            id="file-input"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="file-input">
            <Button variant="contained" component="span">
              파일 업로드
            </Button>
          </label>
          <Typography variant="body1">{uploadedFileName}</Typography>

          <TextField
            multiline
            fullWidth
            placeholder="텍스트를 입력하세요."
            variant="outlined"
            InputProps={{
              style: styles.input,
              endAdornment: (
                <InputAdornment position="end">
                </InputAdornment>
              ),
            }}
          />
          <Button variant="contained" onClick={handleSend}>
                  전송
                </Button>

          {filePreview && (
            <Box sx={{ maxHeight: '400px', overflowY: 'auto', overflowX: 'hidden' }}>
              <Typography variant="h6">파일 미리보기</Typography>
              <pre>{XLSX.stringify(filePreview, null, 2)}</pre>
            </Box>
          )}

          
        </Grid>
      </Grid>
    </Grid>
  );
};

export default DashboardDefaultCopy;
