import axios from 'axios';

const axiosBase = axios.create({
  baseURL : 'http://43.201.55.212:8000/' // http://127.0.0.1:8000
})

export default axiosBase;