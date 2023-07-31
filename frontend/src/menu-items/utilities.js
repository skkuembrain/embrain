// assets
import {
  AppstoreAddOutlined,
  AntDesignOutlined,
  BarcodeOutlined,
  BgColorsOutlined,
  FontSizeOutlined,
  LoadingOutlined
} from '@ant-design/icons';

// icons
const icons = {
  FontSizeOutlined,
  BgColorsOutlined,
  BarcodeOutlined,
  AntDesignOutlined,
  LoadingOutlined,
  AppstoreAddOutlined
};

// ==============================|| MENU ITEMS - UTILITIES ||============================== //

const utilities = {
  id: 'utilities',
  title: 'Model',
  type: 'group',
  children: [
    {
      id: 'util-typography',
      title: 'koGPT',
      type: 'item',
      url: '/typography',
      icon: icons.FontSizeOutlined
    },
    {
      id: 'util-color',
      title: 'koGPT2',
      type: 'item',
      url: '/color',
      icon: icons.BgColorsOutlined
    },
    {
      id: 'util-shadow',
      title: 'polyglotKo',
      type: 'item',
      url: '/shadow',
      icon: icons.BarcodeOutlined
    },
    {
      id: 'ant-icons',
      title: 'trinity',
      type: 'item',
      url: '/icons/ant',
      icon: icons.AntDesignOutlined,
      breadcrumbs: false
    }
  ]
};

export default utilities;
