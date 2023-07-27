import React from 'react'

// import { Helmet } from 'react-helmet'

import './frame-sum.css'

import logo from "assets/external/embrainlogo-200h-200h.png";

const FrameSum = (props) => {
  return (
    <div className="frame-sum-container">
        <title>exported project</title>
      <div className="frame-sum-frame">
        <div className="frame-sum-user-input">
          <div className="frame-sum-task-option">
            <div className="frame-sum-button">
              <div className="frame-sum-summary-button button">
                <input
                  type="radio"
                  name="radio"
                  checked
                  className="frame-sum-radiobutton"
                />
                <span className="frame-sum-text">Summary</span>
              </div>
              <div className="frame-sum-s-button button">
                <input
                  type="radio"
                  name="radio"
                  className="frame-sum-radiobutton1"
                />
                <span className="frame-sum-text01">Sentiment Analysis</span>
              </div>
              <div className="frame-sum-keyword-button button">
                <input
                  type="radio"
                  name="radio"
                  className="frame-sum-radiobutton2"
                />
                <span className="frame-sum-text02">
                  <span>Keyword extraction</span>
                  <br></br>
                </span>
              </div>
            </div>
          </div>
          <div className="frame-sum-input">
            <div className="frame-sum-input-box">
              <div className="frame-sum-inner-input-box">
                <div className="frame-sum-user-link-input">
                  <svg viewBox="0 0 1024 1024" className="frame-sum-icon">
                    <path d="M884.608 441.301l-392.107 392.107c-41.685 41.685-96.256 62.507-150.955 62.507s-109.269-20.821-150.955-62.507-62.507-96.256-62.507-150.955 20.821-109.269 62.507-150.955l392.107-392.107c25.003-25.003 57.728-37.504 90.581-37.504s65.536 12.501 90.581 37.504 37.504 57.728 37.504 90.581-12.501 65.536-37.504 90.581l-392.533 392.107c-8.363 8.363-19.243 12.544-30.208 12.544s-21.845-4.181-30.208-12.501-12.501-19.2-12.501-30.208 4.181-21.845 12.501-30.208l362.24-361.813c16.683-16.64 16.683-43.648 0.043-60.331s-43.648-16.683-60.331-0.043l-362.24 361.813c-25.003 25.003-37.504 57.856-37.504 90.539s12.501 65.536 37.504 90.539 57.856 37.504 90.539 37.504 65.536-12.501 90.539-37.504l392.533-392.107c41.685-41.685 62.507-96.341 62.507-150.912s-20.864-109.269-62.507-150.912-96.341-62.507-150.912-62.507-109.269 20.864-150.912 62.507l-392.107 392.107c-58.325 58.325-87.509 134.869-87.509 211.285s29.184 152.96 87.509 211.285 134.869 87.509 211.285 87.509 152.96-29.184 211.285-87.509l392.107-392.107c16.683-16.683 16.683-43.691 0-60.331s-43.691-16.683-60.331 0z"></path>
                  </svg>
                </div>
                <textarea
                  placeholder="Generate message.."
                  autoFocus
                  className="frame-sum-textarea textarea"
                ></textarea>
                <svg
                  viewBox="0 0 1024.5851428571427 1024"
                  className="frame-sum-icon02"
                >
                  <path d="M1008 6.286c12 8.571 17.714 22.286 15.429 36.571l-146.286 877.714c-1.714 10.857-8.571 20-18.286 25.714-5.143 2.857-11.429 4.571-17.714 4.571-4.571 0-9.143-1.143-13.714-2.857l-301.143-122.857-170.286 186.857c-6.857 8-16.571 12-26.857 12-4.571 0-9.143-0.571-13.143-2.286-14.286-5.714-23.429-19.429-23.429-34.286v-258.286l-269.714-110.286c-13.143-5.143-21.714-17.143-22.857-31.429-1.143-13.714 6.286-26.857 18.286-33.714l950.857-548.571c12-7.429 27.429-6.857 38.857 1.143zM812.571 862.857l126.286-756-819.429 472.571 192 78.286 493.143-365.143-273.143 455.429z"></path>
                </svg>
              </div>
            </div>
            <div className="frame-sum-cancel-guide">
              <span className="frame-sum-text-cancel-guide">
                <span>ESC or Click to cancel</span>
              </span>
            </div>
          </div>
        </div>
        <div className="frame-sum-dialog">
          <div className="frame-sum-input1">
            <div className="frame-sum-date-user-name">
              <div className="frame-sum-date-user-nameframe">
                <span className="frame-sum-text-date">
                  <span>2.03 PM, 15 Nov</span>
                </span>
                <span className="frame-sum-text-user-name">User-1</span>
              </div>
            </div>
            <div className="frame-sum-users-input-data">
              <div className="frame-sum-users-input-data1">
                <svg
                  viewBox="0 0 877.7142857142857 1024"
                  className="frame-sum-icon04"
                >
                  <path d="M838.857 217.143c21.143 21.143 38.857 63.429 38.857 93.714v658.286c0 30.286-24.571 54.857-54.857 54.857h-768c-30.286 0-54.857-24.571-54.857-54.857v-914.286c0-30.286 24.571-54.857 54.857-54.857h512c30.286 0 72.571 17.714 93.714 38.857zM585.143 77.714v214.857h214.857c-3.429-9.714-8.571-19.429-12.571-23.429l-178.857-178.857c-4-4-13.714-9.143-23.429-12.571zM804.571 950.857v-585.143h-237.714c-30.286 0-54.857-24.571-54.857-54.857v-237.714h-438.857v877.714h731.429zM245.143 817.143v60.571h160.571v-60.571h-42.857l58.857-92c6.857-10.857 10.286-19.429 12-19.429h1.143c0.571 2.286 1.714 4 2.857 5.714 2.286 4.571 5.714 8 9.714 13.714l61.143 92h-43.429v60.571h166.286v-60.571h-38.857l-109.714-156 111.429-161.143h38.286v-61.143h-159.429v61.143h42.286l-58.857 90.857c-6.857 10.857-12 19.429-12 18.857h-1.143c-0.571-2.286-1.714-4-2.857-5.714-2.286-4-5.143-8-9.714-13.143l-60.571-90.857h43.429v-61.143h-165.714v61.143h38.857l108 155.429-110.857 161.714h-38.857z"></path>
                </svg>
                <span className="frame-sum-what-user-text">
                  센카 클렌징 U&amp;A 조사 데이터.xlsx
                </span>
                <span className="frame-sum-user-file-volume">
                  <span>2 mb</span>
                </span>
              </div>
              <div className="frame-sum-re-download-button button">
                <svg
                  viewBox="0 0 950.8571428571428 1024"
                  className="frame-sum-icon-download"
                >
                  <path d="M731.429 768c0-20-16.571-36.571-36.571-36.571s-36.571 16.571-36.571 36.571 16.571 36.571 36.571 36.571 36.571-16.571 36.571-36.571zM877.714 768c0-20-16.571-36.571-36.571-36.571s-36.571 16.571-36.571 36.571 16.571 36.571 36.571 36.571 36.571-16.571 36.571-36.571zM950.857 640v182.857c0 30.286-24.571 54.857-54.857 54.857h-841.143c-30.286 0-54.857-24.571-54.857-54.857v-182.857c0-30.286 24.571-54.857 54.857-54.857h265.714l77.143 77.714c21.143 20.571 48.571 32 77.714 32s56.571-11.429 77.714-32l77.714-77.714h265.143c30.286 0 54.857 24.571 54.857 54.857zM765.143 314.857c5.714 13.714 2.857 29.714-8 40l-256 256c-6.857 7.429-16.571 10.857-25.714 10.857s-18.857-3.429-25.714-10.857l-256-256c-10.857-10.286-13.714-26.286-8-40 5.714-13.143 18.857-22.286 33.714-22.286h146.286v-256c0-20 16.571-36.571 36.571-36.571h146.286c20 0 36.571 16.571 36.571 36.571v256h146.286c14.857 0 28 9.143 33.714 22.286z"></path>
                </svg>
                <span className="frame-sum-text-download">
                  <span>Download</span>
                </span>
              </div>
            </div>
          </div>
          <div className="frame-sum-output">
            <div className="frame-sum-embrain-model">
              <div className="frame-sum-logo-embrain">
                <img
                  alt="unnamed932w7rR4ftransformed11226"
                  src={logo}
                  className="frame-sum-logo-embrain1"
                />
              </div>
              <div className="frame-sum-frame-output-date">
                <span className="frame-sum-text-output-date">
                  <span>2.03 PM, 15 Nov</span>
                </span>
              </div>
            </div>
            <div className="frame-sum-model-output-data">
              <div className="frame-sum-frame-output-data">
                <svg
                  viewBox="0 0 877.7142857142857 1024"
                  className="frame-sum-icon07"
                >
                  <path d="M838.857 217.143c21.143 21.143 38.857 63.429 38.857 93.714v658.286c0 30.286-24.571 54.857-54.857 54.857h-768c-30.286 0-54.857-24.571-54.857-54.857v-914.286c0-30.286 24.571-54.857 54.857-54.857h512c30.286 0 72.571 17.714 93.714 38.857zM585.143 77.714v214.857h214.857c-3.429-9.714-8.571-19.429-12.571-23.429l-178.857-178.857c-4-4-13.714-9.143-23.429-12.571zM804.571 950.857v-585.143h-237.714c-30.286 0-54.857-24.571-54.857-54.857v-237.714h-438.857v877.714h731.429zM245.143 817.143v60.571h160.571v-60.571h-42.857l58.857-92c6.857-10.857 10.286-19.429 12-19.429h1.143c0.571 2.286 1.714 4 2.857 5.714 2.286 4.571 5.714 8 9.714 13.714l61.143 92h-43.429v60.571h166.286v-60.571h-38.857l-109.714-156 111.429-161.143h38.286v-61.143h-159.429v61.143h42.286l-58.857 90.857c-6.857 10.857-12 19.429-12 18.857h-1.143c-0.571-2.286-1.714-4-2.857-5.714-2.286-4-5.143-8-9.714-13.143l-60.571-90.857h43.429v-61.143h-165.714v61.143h38.857l108 155.429-110.857 161.714h-38.857z"></path>
                </svg>
                <span className="frame-sum-text10">
                  센카 클렌징 U&amp;A 조사 데이터_result.xlsx
                </span>
                <span className="frame-sum-text11">
                  <span>2 mb</span>
                </span>
              </div>
              <div className="frame-sum-download-button button">
                <svg
                  viewBox="0 0 950.8571428571428 1024"
                  className="frame-sum-icon-download-button"
                >
                  <path d="M731.429 768c0-20-16.571-36.571-36.571-36.571s-36.571 16.571-36.571 36.571 16.571 36.571 36.571 36.571 36.571-16.571 36.571-36.571zM877.714 768c0-20-16.571-36.571-36.571-36.571s-36.571 16.571-36.571 36.571 16.571 36.571 36.571 36.571 36.571-16.571 36.571-36.571zM950.857 640v182.857c0 30.286-24.571 54.857-54.857 54.857h-841.143c-30.286 0-54.857-24.571-54.857-54.857v-182.857c0-30.286 24.571-54.857 54.857-54.857h265.714l77.143 77.714c21.143 20.571 48.571 32 77.714 32s56.571-11.429 77.714-32l77.714-77.714h265.143c30.286 0 54.857 24.571 54.857 54.857zM765.143 314.857c5.714 13.714 2.857 29.714-8 40l-256 256c-6.857 7.429-16.571 10.857-25.714 10.857s-18.857-3.429-25.714-10.857l-256-256c-10.857-10.286-13.714-26.286-8-40 5.714-13.143 18.857-22.286 33.714-22.286h146.286v-256c0-20 16.571-36.571 36.571-36.571h146.286c20 0 36.571 16.571 36.571 36.571v256h146.286c14.857 0 28 9.143 33.714 22.286z"></path>
                </svg>
                <span className="frame-sum-text-download-button">
                  <span>Download</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FrameSum
