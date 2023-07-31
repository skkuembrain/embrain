// project import
import Routes from 'routes';
import ThemeCustomization from 'themes';
import ScrollTop from 'components/ScrollTop';
import Sidebar from 'views/sidebar';


// ==============================|| APP - THEME, ROUTER, LOCAL  ||============================== //

const App = () => (
  <ThemeCustomization>
    <ScrollTop>
      <Routes>
      </Routes>   
    </ScrollTop>
  </ThemeCustomization>
);

export default App;
