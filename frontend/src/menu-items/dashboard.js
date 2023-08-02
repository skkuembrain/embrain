// assets
import { FormOutlined } from '@ant-design/icons';

// icons
const icons = {
  FormOutlined
};

// ==============================|| MENU ITEMS - DASHBOARD ||============================== //

const dashboard = {
  id: 'group-dashboard',
  title: 'Navigation',
  type: 'group',
  children: [
    {
      id: 'dashboard',
      title: '분석',
      type: 'item',
      url: '/dashboard/default',
      icon: icons.FormOutlined,
      breadcrumbs: false
    },
  ]

};

export default dashboard;
