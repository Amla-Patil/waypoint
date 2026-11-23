import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import MatchPage from './pages/MatchPage'
import CoursesPage from './pages/CoursesPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="match" element={<MatchPage />} />
        <Route path="courses" element={<CoursesPage />} />
      </Route>
    </Routes>
  )
}
