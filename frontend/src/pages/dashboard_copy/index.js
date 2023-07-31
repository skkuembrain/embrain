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
import Frame159 from 'views_copy/frame-sum'


// ==============================|| DASHBOARD - DEFAULT ||============================== //

const DashboardDefaultCopy = () => {
  const [value, setValue] = useState('today');
  const [slot, setSlot] = useState('week');

  return (

  <Grid container rowSpacing={4.5} columnSpacing={2.75} alignItems="center">
    <Grid item xs={12} md={1} lg={12}>
      <Grid container alignItems="center" justifyContent="space-between">
        <Frame159/>
      </Grid>
    </Grid>
  </Grid>

  );
};

export default DashboardDefaultCopy;
