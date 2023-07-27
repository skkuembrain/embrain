import { useState } from 'react';

import { FormControl, InputAdornment, OutlinedInput } from '@mui/material';

// material-ui
import {
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
  Typography
} from '@mui/material';

// project import
import OrdersTable from './OrdersTable';
import IncomeAreaChart from './IncomeAreaChart';
import MonthlyBarChart from './MonthlyBarChart';
import ReportAreaChart from './ReportAreaChart';
import SalesColumnChart from './SalesColumnChart';
import MainCard from 'components/MainCard';
import AnalyticEcommerce from 'components/cards/statistics/AnalyticEcommerce';

// assets
import { GiftOutlined, MessageOutlined, SettingOutlined } from '@ant-design/icons';


// avatar style
const avatarSX = {
  width: 36,
  height: 36,
  fontSize: '1rem'
};

// action style
const actionSX = {
  mt: 0.75,
  ml: 1,
  top: 'auto',
  right: 'auto',
  alignSelf: 'flex-start',
  transform: 'none'
};

// sales report status
const status = [
  {
    value: 'today',
    label: 'Descending'
  },
  {
    value: 'month',
    label: 'Ascending'
  },
  {
    value: 'year',
    label: 'This Year'
  }
];

// figma design
import 'style.css'
import Sidebar from 'views/sidebar'
import Frame159 from 'views/frame-sum'


// ==============================|| DASHBOARD - DEFAULT ||============================== //

const DashboardDefault = () => {
  const [value, setValue] = useState('today');
  const [slot, setSlot] = useState('week');

  return (

    <Grid container rowSpacing={4.5} columnSpacing={2.75} alignItems="center">

      {/* row 1
      <Grid item xs={12} sx={{ mb: -2.25 }}>
        <Typography variant="h5">Task</Typography>
      </Grid>
    
      <Grid item md={8} sx={{ display: { sm: 'none', md: 'block', lg: 'none' } }} />

      
      <Box sx={{ width: '100%', ml: { xs: 0, md: 2.75 }, display: 'flex' }}>
    <Box sx={{ flex: 2, textAlign: 'center' }}>
      <FormControl sx={{ width: '100%' }}>
        <OutlinedInput
          size="small"
          id="header-search"
          startAdornment={
            <InputAdornment position="start" sx={{ mr: -0.5 }}></InputAdornment>
          }
          aria-describedby="header-search-text"
          inputProps={{
            'aria-label': 'weight'
          }}
          placeholder="Text를 입력해주세요 .."
        />
      </FormControl>
    </Box>
    <Box sx={{ flex: 2, textAlign: 'center' }}>
      <FormControl sx={{ width: '100%' }}>
        <OutlinedInput
          size="small"
          id="header-search"
          startAdornment={
            <InputAdornment position="start" sx={{ mr: -0.5 }}></InputAdornment>
          }
          aria-describedby="header-search-text"
          inputProps={{
            'aria-label': 'weight'
          }}
          placeholder="Text를 입력해주세요 .."
        />
      </FormControl>
    </Box>
  </Box> */}
  
  <Grid item xs={12} md={1} lg={12}>
    <Grid container alignItems="center" justifyContent="space-between">
      <Frame159/>
    </Grid>
  </Grid>



  

      {/* row 4 : Open Coding Graph */}
  <Grid item xs={12} md={10} lg={12}>
    <Grid container alignItems="center" justifyContent="space-between">
      <Grid item>
        <Typography variant="h5">Result (Open Coding)</Typography>
      </Grid>
      <Grid item>
        <TextField
          id="standard-select-currency"
          size="small"
          select
          value={value}
          onChange={(e) => setValue(e.target.value)}
          sx={{ '& .MuiInputBase-input': { py: 0.5, fontSize: '0.875rem' } }}
        >
          {status.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>
      </Grid>
    </Grid>
    <MainCard sx={{ mt: 1.75 }}>
      <Stack spacing={1.5} sx={{ mb: -12 }}>
        <Typography variant="h6" color="secondary">
          Net Profit
        </Typography>
        <Typography variant="h4">$1560</Typography>
      </Stack>
      <SalesColumnChart />
    </MainCard>
  </Grid>



</Grid>

  );
};

export default DashboardDefault;
